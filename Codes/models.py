import math
from typing import List, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepchem.models.losses import Loss, L2Loss
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.optimizers import Optimizer, LearningRateSchedule

from layers import CustomPositionwiseFeedForward
from loss import LogitAdjASLTopPush
from openpom.utils.optimizer import get_optimizer

try:
    import dgl
    from dgl import DGLGraph
    from dgl.nn.pytorch import Set2Set
    from dgllife.model.gnn import MPNNGNN
except (ImportError, ModuleNotFoundError):
    raise ImportError('This module requires dgl and dgllife')


class MultiHeadGlobalAttentionLite(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 6,
                 gate_mode: str = "softmax", gate_temp: float = 1.2,
                 add_residual_mean: bool = True, add_residual_max: bool = True,
                 use_head_scale: bool = True):
        super().__init__()
        assert gate_mode in ("softmax", "sigmoid")
        self.num_heads = num_heads
        self.gate_mode = gate_mode
        self.gate_temp = gate_temp
        self.add_residual_mean = add_residual_mean
        self.add_residual_max = add_residual_max
        self.use_head_scale = use_head_scale

        base = num_heads * input_dim
        if add_residual_mean:
            base += input_dim
        if add_residual_max:
            base += input_dim
        self.output_dim = base

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_heads)
        ])
        if use_head_scale:
            self.head_scale = nn.Parameter(torch.ones(num_heads))
        else:
            self.register_parameter('head_scale', None)

    def forward(self, g, node_feats: torch.Tensor) -> torch.Tensor:
        with g.local_scope():
            x = node_feats
            g.ndata['x'] = x
            outs = []
            eps = 1e-8

            for h, gate_net in enumerate(self.gates):
                logits = gate_net(x) / self.gate_temp
                if self.gate_mode == "softmax":
                    g.ndata['gate_logit'] = logits
                    max_gate = dgl.max_nodes(g, 'gate_logit')
                    g.ndata['gate_exp'] = torch.exp(logits - dgl.broadcast_nodes(g, max_gate))
                    sum_gate = dgl.sum_nodes(g, 'gate_exp')
                    weights = g.ndata['gate_exp'] / (dgl.broadcast_nodes(g, sum_gate) + eps)
                    g.ndata['xw'] = weights * x
                    pooled = dgl.sum_nodes(g, 'xw')
                else:
                    gate = torch.sigmoid(logits)
                    g.ndata['gate'] = gate
                    g.ndata['xw'] = gate * x
                    sum_xw = dgl.sum_nodes(g, 'xw')
                    sum_g = dgl.sum_nodes(g, 'gate')
                    pooled = sum_xw / (sum_g + eps)

                if self.use_head_scale:
                    pooled = self.head_scale[h] * pooled
                outs.append(pooled)

            H = torch.cat(outs, dim=1)
            if self.add_residual_mean:
                meanv = dgl.mean_nodes(g, 'x')
                H = torch.concat([H, meanv], dim=1)
            if self.add_residual_max:
                maxv = dgl.max_nodes(g, 'x')
                H = torch.concat([H, maxv], dim=1)
            return H


class FamilyPromptHyperLoRAHead(nn.Module):
    """
    Reliability-, hub-aware competitor-, and cardinality-aware family head.

    Motivation from diagnostics:
    - competitor-aware calibration already improves macro PR by rescuing tail
      labels from being overshadowed by frequent co-occurring labels;
    - however, the gain mainly comes from mid/tail buckets while low-cardinality
      samples (especially k=1/2) still underperform, indicating the rescue is
      too aggressive for sparse-label molecules.

    Design:
    - keep the sample-conditioned delta logits as the main fine-grained
      correction path;
    - estimate both a generic competition gap and a hub-weighted competition
      route, because the worst tail labels are repeatedly suppressed by a small
      set of high-frequency hub labels;
    - modulate rescue strength by family dispersion and a self-estimated soft
      cardinality context, so sparse-label samples receive gentler rescue while
      ultra-tail labels under hub suppression still get targeted correction.
    """

    def __init__(self,
                 n_tasks: int,
                 graph_dim: int,
                 embed_dim: int,
                 attn_dim: int,
                 s2s_dim: int,
                 label_family_prior,
                 class_imbalance_ratio: Optional[List[float]] = None,
                 prompt_dim: int = 128,
                 prompt_heads: int = 4,
                 adapter_rank: int = 16,
                 adapter_dropout: float = 0.08,
                 delta_scale_init: float = 0.025,
                 bias_scale_init: float = 0.035,
                 label_competitor_prior=None,
                 label_hub_score=None,
                 competitor_delta_gain: float = 0.65,
                 competitor_bias_suppress: float = 0.75,
                 competitor_floor: float = 0.40,
                 diffuse_tail_gain: float = 0.18,
                 family_reliability_min: float = 0.65,
                 label_cardinality_center: float = 4.0,
                 label_cardinality_scale: float = 1.5,
                 cardinality_blend: float = 0.60,
                 cardinality_min_gate: float = 0.35,
                 hub_delta_gain: float = 0.32,
                 hub_bias_suppress: float = 0.22,
                 ultra_tail_threshold: float = 6.0,
                 hub_competitor_mix: float = 0.55,
                 hub_self_margin: float = 0.85):
        super().__init__()
        prior = torch.as_tensor(label_family_prior, dtype=torch.float32)
        if prior.dim() != 2 or prior.size(0) != n_tasks:
            raise ValueError("label_family_prior must have shape [n_tasks, n_families]")
        self.n_tasks = n_tasks
        self.n_families = int(prior.size(1))
        self.prompt_dim = int(prompt_dim)
        self.adapter_rank = int(adapter_rank)

        self.register_buffer('label_family_prior', prior)
        self.register_buffer('label_family_peak', prior.max(dim=1).values)
        if class_imbalance_ratio is None:
            r = torch.ones(n_tasks, dtype=torch.float32)
            freq_scale = torch.ones(n_tasks, dtype=torch.float32)
            tail_strength = torch.zeros(n_tasks, dtype=torch.float32)
            ultra_tail_strength = torch.zeros(n_tasks, dtype=torch.float32)
        else:
            r = torch.as_tensor(class_imbalance_ratio, dtype=torch.float32).clamp(min=1e-6)
            tail = torch.sigmoid(2.0 * (torch.log(r) - math.log(3.0)))
            ultra_tail = torch.sigmoid(2.4 * (torch.log(r) - math.log(float(max(ultra_tail_threshold, 1.0001)))))
            freq_scale = 0.35 + 0.65 * tail
            tail_strength = tail
            ultra_tail_strength = ultra_tail
        self.register_buffer('label_freq_scale', freq_scale)
        self.register_buffer('tail_strength', tail_strength)
        self.register_buffer('ultra_tail_strength', ultra_tail_strength)
        self.register_buffer('class_imbalance_ratio_tensor', r)

        if label_competitor_prior is None:
            comp = torch.zeros((n_tasks, n_tasks), dtype=torch.float32)
        else:
            comp = torch.as_tensor(label_competitor_prior, dtype=torch.float32)
            if comp.dim() != 2 or comp.shape != (n_tasks, n_tasks):
                raise ValueError("label_competitor_prior must have shape [n_tasks, n_tasks]")
        self.register_buffer('label_competitor_prior', comp)

        if label_hub_score is None:
            hub_score = torch.zeros(n_tasks, dtype=torch.float32)
        else:
            hub_score = torch.as_tensor(label_hub_score, dtype=torch.float32)
            if hub_score.dim() != 1 or hub_score.shape[0] != n_tasks:
                raise ValueError("label_hub_score must have shape [n_tasks]")
        hub_score = hub_score.clamp(min=0.0, max=1.0)
        self.register_buffer('label_hub_score', hub_score)
        hub_comp = comp * hub_score.unsqueeze(0)
        hub_comp = hub_comp / hub_comp.sum(dim=1, keepdim=True).clamp(min=1e-8)
        self.register_buffer('label_hub_competitor_prior', hub_comp)

        self.competitor_delta_gain = float(competitor_delta_gain)
        self.competitor_bias_suppress = float(competitor_bias_suppress)
        self.competitor_floor = float(competitor_floor)
        self.diffuse_tail_gain = float(diffuse_tail_gain)
        self.family_reliability_min = float(family_reliability_min)
        self.label_cardinality_center = float(label_cardinality_center)
        self.label_cardinality_scale = float(max(label_cardinality_scale, 1e-3))
        self.cardinality_blend = float(cardinality_blend)
        self.cardinality_min_gate = float(cardinality_min_gate)
        self.hub_delta_gain = float(hub_delta_gain)
        self.hub_bias_suppress = float(hub_bias_suppress)
        self.hub_competitor_mix = float(hub_competitor_mix)
        self.hub_self_margin = float(hub_self_margin)

        self.family_queries = nn.Parameter(torch.randn(self.n_families, prompt_dim) * 0.02)
        self.family_anchors = nn.Parameter(torch.randn(self.n_families, prompt_dim) * 0.02)
        self.label_prototypes = nn.Parameter(torch.randn(n_tasks, prompt_dim) * 0.02)
        self.label_bias = nn.Parameter(torch.zeros(n_tasks))

        self.delta_scale = nn.Parameter(torch.tensor(float(delta_scale_init), dtype=torch.float32))
        self.bias_scale = nn.Parameter(torch.tensor(float(bias_scale_init), dtype=torch.float32))
        self.support_gate_delta = nn.Parameter(torch.full((n_tasks,), -1.75, dtype=torch.float32))

        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, prompt_dim),
            nn.LayerNorm(prompt_dim)
        )
        self.attn_token_proj = nn.Sequential(
            nn.Linear(attn_dim, prompt_dim),
            nn.LayerNorm(prompt_dim)
        )
        self.s2s_token_proj = nn.Sequential(
            nn.Linear(s2s_dim, prompt_dim),
            nn.LayerNorm(prompt_dim)
        )

        self.query_attn = nn.MultiheadAttention(prompt_dim, prompt_heads, dropout=adapter_dropout, batch_first=True)
        self.ctx_norm1 = nn.LayerNorm(prompt_dim)
        self.ctx_ffn = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim * 2),
            nn.SiLU(),
            nn.Dropout(adapter_dropout),
            nn.Linear(prompt_dim * 2, prompt_dim)
        )
        self.ctx_norm2 = nn.LayerNorm(prompt_dim)

        self.sample_factor = nn.Sequential(
            nn.Linear(embed_dim + prompt_dim, prompt_dim),
            nn.SiLU(),
            nn.Dropout(adapter_dropout),
            nn.Linear(prompt_dim, adapter_rank)
        )
        self.label_factor = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
            nn.SiLU(),
            nn.Linear(prompt_dim, adapter_rank)
        )
        self.gate_query = nn.Sequential(
            nn.Linear(embed_dim + prompt_dim, prompt_dim),
            nn.SiLU(),
            nn.Linear(prompt_dim, prompt_dim)
        )
        self.gate_label = nn.Linear(prompt_dim, prompt_dim, bias=False)

        self.proto_graph_proj = nn.Linear(prompt_dim, prompt_dim)
        self.proto_label_proj = nn.Linear(prompt_dim, prompt_dim)
        self.family_graph_proj = nn.Linear(prompt_dim, prompt_dim)
        self.family_label_proj = nn.Linear(prompt_dim, prompt_dim)
        self.label_seed_norm = nn.LayerNorm(prompt_dim)

    def _build_family_summary(self,
                              graph_feat: torch.Tensor,
                              attn_feat: torch.Tensor,
                              s2s_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.stack([
            self.graph_proj(graph_feat),
            self.attn_token_proj(attn_feat),
            self.s2s_token_proj(s2s_feat),
        ], dim=1)
        q = self.family_queries.unsqueeze(0).expand(tokens.size(0), -1, -1)
        ctx, _ = self.query_attn(q, tokens, tokens, need_weights=False)
        ctx = self.ctx_norm1(ctx + q)
        ctx = self.ctx_norm2(ctx + self.ctx_ffn(ctx))

        anchor = F.normalize(self.family_anchors, dim=-1)
        fam_logits = (F.normalize(ctx, dim=-1) * anchor.unsqueeze(0)).sum(dim=-1) * math.sqrt(self.prompt_dim)
        fam_weights = torch.softmax(fam_logits, dim=-1)
        family_summary = (fam_weights.unsqueeze(-1) * ctx).sum(dim=1)
        return family_summary, fam_weights

    def forward(self,
                embeddings: torch.Tensor,
                graph_feat: torch.Tensor,
                attn_feat: torch.Tensor,
                s2s_feat: torch.Tensor,
                base_logits: torch.Tensor) -> torch.Tensor:
        family_summary, fam_weights = self._build_family_summary(graph_feat, attn_feat, s2s_feat)

        label_seed = self.label_prototypes + self.label_family_prior @ self.family_anchors
        label_seed = self.label_seed_norm(label_seed)

        sample_feat = torch.cat([embeddings, family_summary], dim=-1)
        sample_factor = F.normalize(self.sample_factor(sample_feat), dim=-1)
        label_factor = F.normalize(self.label_factor(label_seed), dim=-1)
        delta_logits = sample_factor @ label_factor.t()

        proto_graph = F.normalize(self.proto_graph_proj(family_summary), dim=-1)
        proto_label = F.normalize(self.proto_label_proj(label_seed), dim=-1)
        proto_logits = proto_graph @ proto_label.t()
        delta_logits = 0.7 * delta_logits + 0.3 * proto_logits

        gate_query = F.normalize(self.gate_query(sample_feat), dim=-1)
        gate_label = F.normalize(self.gate_label(label_seed), dim=-1)
        gate_logits = gate_query @ gate_label.t() / math.sqrt(self.prompt_dim)
        base_uncertainty = 1.0 - torch.tanh(base_logits.abs() / 2.0)
        gate = torch.sigmoid(gate_logits + self.support_gate_delta.unsqueeze(0))
        gate = gate * self.label_freq_scale.unsqueeze(0) * base_uncertainty

        family_label_logits = F.normalize(self.family_graph_proj(family_summary), dim=-1) @             F.normalize(self.family_label_proj(label_seed), dim=-1).t()
        family_bias = fam_weights @ self.label_family_prior.t()
        family_bias = family_bias - (1.0 / max(self.n_families, 1))
        family_bias = 0.5 * family_bias + 0.5 * family_label_logits

        # Competitor-aware tail calibration with hub-aware selective rescue.
        eps = 1e-8
        base_prob = torch.sigmoid(base_logits.detach())
        competitor_pressure = (base_prob @ self.label_competitor_prior.t()).clamp(min=0.0, max=1.0)
        hub_pressure = (base_prob @ self.label_hub_competitor_prior.t()).clamp(min=0.0, max=1.0)
        competition_gap = F.relu(competitor_pressure - base_prob)
        hub_gap = F.relu(hub_pressure - self.hub_self_margin * base_prob)

        uncertain_tail = base_uncertainty * self.tail_strength.unsqueeze(0)
        diffuse_tail = uncertain_tail * (1.0 - self.label_family_peak.unsqueeze(0))
        ultra_tail = uncertain_tail * self.ultra_tail_strength.unsqueeze(0)

        fam_entropy = -(fam_weights.clamp(min=eps) * torch.log(fam_weights.clamp(min=eps))).sum(dim=1, keepdim=True)
        fam_entropy = fam_entropy / math.log(float(max(self.n_families, 2)))

        soft_card = base_prob.sum(dim=1, keepdim=True)
        card_context = torch.sigmoid((soft_card - self.label_cardinality_center) / self.label_cardinality_scale)
        rescue_context = self.cardinality_blend * card_context + (1.0 - self.cardinality_blend) * fam_entropy
        rescue_context = self.cardinality_min_gate + (1.0 - self.cardinality_min_gate) * rescue_context

        lowcard_guard = torch.sigmoid((soft_card - (self.label_cardinality_center - 0.75)) / (0.85 * self.label_cardinality_scale))
        ultra_context = 0.25 + 0.75 * lowcard_guard

        generic_route = competition_gap * uncertain_tail * rescue_context
        hub_route = hub_gap * ultra_tail * ultra_context
        mixed_route = self.hub_competitor_mix * generic_route + (1.0 - self.hub_competitor_mix) * hub_route

        delta_gain = 1.0 + self.competitor_delta_gain * mixed_route + self.hub_delta_gain * hub_route + self.diffuse_tail_gain * diffuse_tail * fam_entropy
        bias_damp = 1.0 - self.competitor_bias_suppress * mixed_route - self.hub_bias_suppress * hub_route
        bias_damp = bias_damp.clamp(min=self.competitor_floor, max=1.0)
        family_reliability = self.family_reliability_min + (1.0 - self.family_reliability_min) * self.label_family_peak.unsqueeze(0)

        final_logits = (
            base_logits
            + torch.tanh(self.delta_scale) * gate * delta_gain * delta_logits
            + torch.tanh(self.bias_scale) * self.label_freq_scale.unsqueeze(0) * family_reliability * bias_damp * family_bias
            + self.label_bias.unsqueeze(0)
        )
        return final_logits


class MPNNPOM(nn.Module):
    def __init__(self,
                 n_tasks: int,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 edge_out_feats: int = 64,
                 num_step_message_passing: int = 3,
                 mpnn_residual: bool = True,
                 message_aggregator_type: str = 'sum',
                 mode: str = 'classification',
                 number_atom_features: int = 134,
                 number_bond_features: int = 6,
                 n_classes: int = 1,
                 nfeat_name: str = 'x',
                 efeat_name: str = 'edge_attr',
                 readout_type: str = 'set2set',
                 num_step_set2set: int = 6,
                 num_layer_set2set: int = 3,
                 ffn_hidden_list: List = [300],
                 ffn_embeddings: int = 256,
                 ffn_activation: str = 'relu',
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True,
                 gate_temp: float = 1.2,
                 num_heads: int = 6,
                 label_family_prior=None,
                 class_imbalance_ratio=None,
                 family_prompt_dim: int = 128,
                 family_prompt_heads: int = 4,
                 family_adapter_rank: int = 16,
                 family_adapter_dropout: float = 0.08,
                 family_delta_scale_init: float = 0.025,
                 family_bias_scale_init: float = 0.035,
                 label_competitor_prior=None,
                 label_hub_score=None,
                 family_competitor_delta_gain: float = 0.65,
                 family_competitor_bias_suppress: float = 0.75,
                 family_competitor_floor: float = 0.40,
                 family_diffuse_tail_gain: float = 0.18,
                 family_reliability_min: float = 0.65,
                 label_cardinality_center: float = 4.0,
                 label_cardinality_scale: float = 1.5,
                 family_cardinality_blend: float = 0.60,
                 family_cardinality_min_gate: float = 0.35,
                 family_hub_delta_gain: float = 0.32,
                 family_hub_bias_suppress: float = 0.22,
                 family_ultra_tail_threshold: float = 6.0,
                 family_hub_competitor_mix: float = 0.55,
                 family_hub_self_margin: float = 0.85):
        super().__init__()
        assert mode in ['classification', 'regression'], "Invalid mode"

        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.mode = mode
        self.nfeat_name = nfeat_name
        self.efeat_name = efeat_name
        self.readout_type = readout_type

        self.atom_proj = nn.Sequential(
            nn.Linear(number_atom_features, 256),
            nn.SiLU(),
            nn.Linear(256, node_out_feats)
        )
        self.atom_norm = nn.LayerNorm(node_out_feats)
        self.atom_drop = nn.Dropout(0.2)

        self.bond_proj = nn.Sequential(
            nn.Linear(number_bond_features, 128),
            nn.SiLU(),
            nn.Linear(128, edge_hidden_feats)
        )
        self.bond_norm = nn.LayerNorm(edge_hidden_feats)
        self.bond_drop = nn.Dropout(0.2)

        self.mpnn = MPNNGNN(
            node_in_feats=node_out_feats,
            edge_in_feats=edge_hidden_feats,
            node_out_feats=node_out_feats,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
        )
        D = node_out_feats + edge_hidden_feats

        if self.readout_type == 'attn_pool':
            self.readout_attn = MultiHeadGlobalAttentionLite(
                input_dim=D,
                hidden_dim=64,
                num_heads=num_heads,
                gate_mode='softmax',
                gate_temp=gate_temp,
                add_residual_mean=True,
                add_residual_max=True,
                use_head_scale=True
            )
            self.readout_set2set = Set2Set(
                input_dim=D,
                n_iters=num_step_set2set,
                n_layers=num_layer_set2set
            )
            self.attn_dim = self.readout_attn.output_dim
            self.s2s_dim = 2 * D
            ffn_input = self.attn_dim + self.s2s_dim
            self.fusion_gate = nn.Sequential(
                nn.Linear(ffn_input, ffn_input // 2),
                nn.SiLU(),
                nn.Linear(ffn_input // 2, ffn_input),
                nn.Sigmoid()
            )
            self.readout_post_ln = nn.LayerNorm(ffn_input)
        elif self.readout_type == 'set2set':
            self.readout_set2set = Set2Set(input_dim=D, n_iters=num_step_set2set, n_layers=num_layer_set2set)
            ffn_input = 2 * D
            self.readout_post_ln = None
            self.attn_dim = None
            self.s2s_dim = 2 * D
        elif self.readout_type == 'global_sum_pooling':
            ffn_input = D
            self.readout_post_ln = None
            self.attn_dim = None
            self.s2s_dim = D
        else:
            raise ValueError("Invalid readout_type")

        d_hidden_list = ffn_hidden_list + [ffn_embeddings] if ffn_embeddings else ffn_hidden_list
        self.ffn = CustomPositionwiseFeedForward(
            d_input=ffn_input,
            d_hidden_list=d_hidden_list,
            d_output=n_tasks * n_classes if mode == 'classification' else n_tasks,
            activation=ffn_activation,
            dropout_p=ffn_dropout_p,
            dropout_at_input_no_act=ffn_dropout_at_input_no_act
        )

        self.label_head = None
        if mode == 'classification' and n_classes == 1 and self.readout_type == 'attn_pool' and label_family_prior is not None:
            self.label_head = FamilyPromptHyperLoRAHead(
                n_tasks=n_tasks,
                graph_dim=ffn_input,
                embed_dim=ffn_embeddings,
                attn_dim=self.attn_dim,
                s2s_dim=self.s2s_dim,
                label_family_prior=label_family_prior,
                class_imbalance_ratio=class_imbalance_ratio,
                prompt_dim=family_prompt_dim,
                prompt_heads=family_prompt_heads,
                adapter_rank=family_adapter_rank,
                adapter_dropout=family_adapter_dropout,
                delta_scale_init=family_delta_scale_init,
                bias_scale_init=family_bias_scale_init,
                label_competitor_prior=label_competitor_prior,
                competitor_delta_gain=family_competitor_delta_gain,
                competitor_bias_suppress=family_competitor_bias_suppress,
                competitor_floor=family_competitor_floor,
                diffuse_tail_gain=family_diffuse_tail_gain,
                family_reliability_min=family_reliability_min,
                label_cardinality_center=label_cardinality_center,
                label_cardinality_scale=label_cardinality_scale,
                cardinality_blend=family_cardinality_blend,
                cardinality_min_gate=family_cardinality_min_gate,
            )

    def _readout(self, g, node_encodings, edge_feats):
        g.ndata['node_emb'] = node_encodings
        g.edata['edge_emb'] = edge_feats

        def message_func(edges):
            src_msg = torch.cat((edges.src['node_emb'], edges.data['edge_emb']), dim=1)
            return {'src_msg': src_msg}

        def reduce_func(nodes):
            return {'src_msg_sum': torch.sum(nodes.mailbox['src_msg'], dim=1)}

        g.send_and_recv(g.edges(), message_func=message_func, reduce_func=reduce_func)

        h = g.ndata['src_msg_sum']
        attn_feat = self.readout_attn(g, h)
        s2s_feat = self.readout_set2set(g, h)

        concat_feat = torch.cat([attn_feat, s2s_feat], dim=1)
        if hasattr(self, 'fusion_gate'):
            gate = self.fusion_gate(concat_feat)
            concat_feat = concat_feat * gate + concat_feat
        return concat_feat, attn_feat, s2s_feat

    def forward(self, g: DGLGraph) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        raw_node_feats = g.ndata[self.nfeat_name]
        raw_edge_feats = g.edata[self.efeat_name]

        node_feats = self.atom_drop(self.atom_norm(self.atom_proj(raw_node_feats)))
        edge_feats = self.bond_drop(self.bond_norm(self.bond_proj(raw_edge_feats)))

        node_encodings = self.mpnn(g, node_feats, edge_feats)
        if self.readout_type == 'attn_pool':
            molecular_encodings, attn_feat, s2s_feat = self._readout(g, node_encodings, edge_feats)
        else:
            raise ValueError("FamilyPromptHyperLoRAHead currently expects readout_type='attn_pool'.")

        if hasattr(self, 'readout_post_ln') and self.readout_post_ln is not None:
            molecular_encodings = self.readout_post_ln(molecular_encodings)
        embeddings, out = self.ffn(molecular_encodings)

        if self.mode == 'classification':
            logits = out.view(-1, self.n_tasks, self.n_classes)
            if self.n_classes == 1:
                base_logits = logits.squeeze(-1)
                if self.label_head is not None:
                    base_logits = self.label_head(embeddings, molecular_encodings, attn_feat, s2s_feat, base_logits)
                proba = torch.sigmoid(base_logits)
                # Important: for binary multi-label DeepChem losses in this project expect
                # loss outputs with shape [batch, n_tasks], not [batch, n_tasks, 1].
                return proba, base_logits, embeddings
            proba = torch.sigmoid(logits)
            return proba, logits, embeddings
        return out


class newModel(TorchModel):
    def __init__(self,
                 n_tasks: int,
                 class_imbalance_ratio: Optional[List] = None,
                 label_family_prior=None,
                 loss_aggr_type: str = 'sum',
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 batch_size: int = 100,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 edge_out_feats: int = 64,
                 num_step_message_passing: int = 3,
                 mpnn_residual: bool = True,
                 message_aggregator_type: str = 'sum',
                 mode: str = 'regression',
                 number_atom_features: int = 134,
                 number_bond_features: int = 6,
                 n_classes: int = 1,
                 readout_type: str = 'set2set',
                 num_step_set2set: int = 6,
                 num_layer_set2set: int = 3,
                 ffn_hidden_list: List = [300],
                 ffn_embeddings: int = 256,
                 ffn_activation: str = 'relu',
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True,
                 weight_decay: float = 1e-5,
                 self_loop: bool = False,
                 optimizer_name: str = 'adam',
                 device_name: Optional[str] = None,
                 gate_temp: float = 1.2,
                 num_heads: int = 6,
                 tau: float = 0.2,
                 toppush_weight: float = 0.035,
                 toppush_margin: float = 0.32,
                 toppush_temp: float = 0.65,
                 toppush_tail_scale: float = 0.85,
                 family_prompt_dim: int = 128,
                 family_prompt_heads: int = 4,
                 family_adapter_rank: int = 16,
                 family_adapter_dropout: float = 0.08,
                 family_delta_scale_init: float = 0.025,
                 family_bias_scale_init: float = 0.035,
                 label_competitor_prior=None,
                 label_hub_score=None,
                 family_competitor_delta_gain: float = 0.65,
                 family_competitor_bias_suppress: float = 0.75,
                 family_competitor_floor: float = 0.40,
                 family_diffuse_tail_gain: float = 0.18,
                 family_reliability_min: float = 0.65,
                 label_cardinality_center: float = 4.0,
                 label_cardinality_scale: float = 1.5,
                 family_cardinality_blend: float = 0.60,
                 family_cardinality_min_gate: float = 0.35,
                 family_hub_delta_gain: float = 0.32,
                 family_hub_bias_suppress: float = 0.22,
                 family_ultra_tail_threshold: float = 6.0,
                 family_hub_competitor_mix: float = 0.55,
                 family_hub_self_margin: float = 0.85,
                 **kwargs):
        model: nn.Module = MPNNPOM(
            n_tasks=n_tasks,
            node_out_feats=node_out_feats,
            edge_hidden_feats=edge_hidden_feats,
            edge_out_feats=edge_out_feats,
            num_step_message_passing=num_step_message_passing,
            mpnn_residual=mpnn_residual,
            message_aggregator_type=message_aggregator_type,
            mode=mode,
            number_atom_features=number_atom_features,
            number_bond_features=number_bond_features,
            n_classes=n_classes,
            readout_type=readout_type,
            num_step_set2set=num_step_set2set,
            num_layer_set2set=num_layer_set2set,
            ffn_hidden_list=ffn_hidden_list,
            ffn_embeddings=ffn_embeddings,
            ffn_activation=ffn_activation,
            ffn_dropout_p=ffn_dropout_p,
            ffn_dropout_at_input_no_act=ffn_dropout_at_input_no_act,
            gate_temp=gate_temp,
            num_heads=num_heads,
            label_family_prior=label_family_prior,
            class_imbalance_ratio=class_imbalance_ratio,
            family_prompt_dim=family_prompt_dim,
            family_prompt_heads=family_prompt_heads,
            family_adapter_rank=family_adapter_rank,
            family_adapter_dropout=family_adapter_dropout,
            family_delta_scale_init=family_delta_scale_init,
            family_bias_scale_init=family_bias_scale_init,
            label_competitor_prior=label_competitor_prior,
            label_hub_score=label_hub_score,
            family_competitor_delta_gain=family_competitor_delta_gain,
            family_competitor_bias_suppress=family_competitor_bias_suppress,
            family_competitor_floor=family_competitor_floor,
            family_diffuse_tail_gain=family_diffuse_tail_gain,
            family_reliability_min=family_reliability_min,
            label_cardinality_center=label_cardinality_center,
            label_cardinality_scale=label_cardinality_scale,
            family_cardinality_blend=family_cardinality_blend,
            family_cardinality_min_gate=family_cardinality_min_gate,
            family_hub_delta_gain=family_hub_delta_gain,
            family_hub_bias_suppress=family_hub_bias_suppress,
            family_ultra_tail_threshold=family_ultra_tail_threshold,
            family_hub_competitor_mix=family_hub_competitor_mix,
            family_hub_self_margin=family_hub_self_margin,
        )

        if class_imbalance_ratio and (len(class_imbalance_ratio) != n_tasks):
            raise Exception("size of class_imbalance_ratio should be equal to n_tasks")

        if mode == 'regression':
            loss: Loss = L2Loss()
            output_types: List = ['prediction']
        else:
            loss = LogitAdjASLTopPush(
                class_imbalance_ratio=class_imbalance_ratio,
                tau=tau,
                gamma_pos=0.0,
                gamma_neg_head=4.0,
                gamma_neg_tail=2.0,
                ir_threshold=3.0,
                clip=0.05,
                reduction="sum",
                toppush_weight=toppush_weight,
                toppush_margin=toppush_margin,
                toppush_temp=toppush_temp,
                toppush_tail_scale=toppush_tail_scale,
                toppush_neg_topk=5,
            )
            output_types = ['prediction', 'loss', 'embedding']

        optimizer: Optimizer = get_optimizer(optimizer_name)
        optimizer.learning_rate = learning_rate
        device = torch.device(device_name) if device_name is not None else None
        super(newModel, self).__init__(
            model,
            loss=loss,
            output_types=output_types,
            optimizer=optimizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=device,
            **kwargs,
        )

        self.weight_decay: float = weight_decay
        self._self_loop: bool = self_loop
        self.regularization_loss = None

    def freeze_for_refine(self,
                          train_ffn: bool = True,
                          train_readout_post_ln: bool = True,
                          train_fusion_gate: bool = False,
                          train_label_head: bool = True):
        for p in self.model.parameters():
            p.requires_grad = False

        train_prefixes = []
        if train_ffn:
            train_prefixes.append('ffn')
        if train_readout_post_ln:
            train_prefixes.append('readout_post_ln')
        if train_fusion_gate:
            train_prefixes.append('fusion_gate')
        if train_label_head:
            train_prefixes.append('label_head')

        for name, p in self.model.named_parameters():
            if any(name.startswith(pref) for pref in train_prefixes):
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def trainable_parameter_names(self) -> List[str]:
        return [name for name, p in self.model.named_parameters() if p.requires_grad]


    def _prepare_batch(self, batch: Tuple[List, List, List]) -> Tuple[DGLGraph, List[torch.Tensor], List[torch.Tensor]]:
        inputs, labels, weights = batch
        dgl_graphs: List[DGLGraph] = [
            graph.to_dgl_graph(self_loop=self._self_loop)
            for graph in inputs[0]
        ]
        g: DGLGraph = dgl.batch(dgl_graphs).to(self.device)
        _, labels, weights = super(newModel, self)._prepare_batch(([], labels, weights))
        return g, labels, weights

    def _regularization_loss(self) -> torch.Tensor:
        l1_regularization: torch.Tensor = torch.tensor(0., requires_grad=True)
        l2_regularization: torch.Tensor = torch.tensor(0., requires_grad=True)
        for name, param in self.model.named_parameters():
            if 'bias' not in name:
                l1_regularization = l1_regularization + torch.norm(param, p=1)
                l2_regularization = l2_regularization + torch.norm(param, p=2)
        l1_norm, l2_norm = 0.0, self.weight_decay
        return l1_norm * l1_regularization + l2_norm * l2_regularization
