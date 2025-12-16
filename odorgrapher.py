import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional, Callable, Dict

from deepchem.models.losses import Loss, L2Loss
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.optimizers import (
    Optimizer, LearningRateSchedule,
    Adam, AdaGrad, AdamW, SparseAdam,
    RMSProp, GradientDescent, KFAC
)

from layers import CustomPositionwiseFeedForward
from loss import AdaptiveLogitAdjustedASL

try:
    import dgl
    from dgl import DGLGraph
    from dgl.nn.pytorch import Set2Set
    from dgllife.model.gnn import MPNNGNN
except (ImportError, ModuleNotFoundError):
    raise ImportError('Requires dgl and dgllife')

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
        self.add_residual_max  = add_residual_max
        self.use_head_scale    = use_head_scale

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
                    sum_g  = dgl.sum_nodes(g, 'gate')                     
                    pooled = sum_xw / (sum_g + eps)

                if self.use_head_scale:
                    pooled = self.head_scale[h] * pooled                 
                outs.append(pooled)

            H = torch.cat(outs, dim=1)                                    
            if self.add_residual_mean:
                meanv = dgl.mean_nodes(g, 'x')                            
                H = torch.concat([H, meanv], dim=1)
            if self.add_residual_max:
                maxv  = dgl.max_nodes(g, 'x')                            
                H = torch.concat([H, maxv], dim=1)
            return H

class OdorGrapher_NN(nn.Module):
    def __init__(self,
                 labels: int,
                 node_out_feats: int = 128,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing = 4,
                 number_atom_features: int = 147,
                 number_bond_features: int = 7,
                 readout: str = 'set2set',
                 num_step_set2set: int = 6,
                 num_layer_set2set: int = 3,
                 ffn_hidden_list: List = [300],
                 ffn_output_dim: int = 256,
                 ffn_activation: str = 'relu',
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True,
                 gate_temp: float = 1.2,
                 num_heads: int = 6):

        super().__init__()

        self.labels = labels
        self.nfeat_name = 'x'
        self.efeat_name = 'edge_attr'
        self.readout = readout

        self.atom_proj = nn.Sequential(
            nn.Linear(number_atom_features, 256),
            nn.SiLU(),
            nn.Linear(256, node_out_feats)
        )
        self.atom_norm = nn.LayerNorm(node_out_feats) 
        self.atom_drop = nn.Dropout(0.05) 

        self.bond_proj = nn.Sequential(
            nn.Linear(number_bond_features, 128),
            nn.SiLU(),
            nn.Linear(128, edge_hidden_feats)
        )

        self.bond_norm = nn.LayerNorm(edge_hidden_feats) 
        self.bond_drop = nn.Dropout(0.05) 

        self.mpnn = MPNNGNN(
            node_in_feats=node_out_feats,
            edge_in_feats=edge_hidden_feats,
            node_out_feats=node_out_feats,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
        )
        D = node_out_feats + edge_hidden_feats

        if self.readout == 'attn_pool':
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
            ffn_input = self.readout_attn.output_dim
            self.readout_post_ln = nn.LayerNorm(ffn_input) 
        elif self.readout == 'set2set':
            self.readout_set2set = Set2Set(input_dim=D, n_iters=num_step_set2set, n_layers=num_layer_set2set)
            ffn_input = 2 * D
            self.readout_post_ln = None
        else:
            raise ValueError("Invalid readout")

        d_hidden_list = ffn_hidden_list + [ffn_output_dim] if ffn_output_dim else ffn_hidden_list
        self.ffn = CustomPositionwiseFeedForward(
            d_input=ffn_input,
            d_hidden_list=d_hidden_list,
            d_output=labels,
            activation=ffn_activation,
            dropout_p=ffn_dropout_p,
            dropout_at_input_no_act=ffn_dropout_at_input_no_act
        )

    def _readout(self, g: DGLGraph, node_encodings: torch.Tensor, edge_feats: torch.Tensor) -> torch.Tensor:
        g.ndata['node_emb'] = node_encodings
        g.edata['edge_emb'] = edge_feats 

        def message_func(edges):
            src_msg = torch.cat((edges.src['node_emb'], edges.data['edge_emb']), dim=1)
            return {'src_msg': src_msg}

        def reduce_func(nodes):
            return {'src_msg_sum': torch.sum(nodes.mailbox['src_msg'], dim=1)}

        g.send_and_recv(g.edges(), message_func=message_func, reduce_func=reduce_func)

        if self.readout == 'set2set':
            return self.readout_set2set(g, g.ndata['src_msg_sum'])
        elif self.readout == 'attn_pool':
            return self.readout_attn(g, g.ndata['src_msg_sum'])
        else:
            return dgl.sum_nodes(g, 'src_msg_sum')

    def forward(self, g: DGLGraph) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        raw_node_feats = g.ndata[self.nfeat_name]
        raw_edge_feats = g.edata[self.efeat_name]

        node_feats = self.atom_drop(self.atom_norm(self.atom_proj(raw_node_feats)))
        edge_feats = self.bond_drop(self.bond_norm(self.bond_proj(raw_edge_feats)))

        node_encodings = self.mpnn(g, node_feats, edge_feats)
        molecular_encodings = self._readout(g, node_encodings, edge_feats)
        if hasattr(self, "readout_post_ln") and self.readout_post_ln is not None:
            molecular_encodings = self.readout_post_ln(molecular_encodings)
        embeddings, out = self.ffn(molecular_encodings)

        logits = out.view(-1, self.labels, 1)
        proba = torch.sigmoid(logits)
        proba = proba.squeeze(-1)
        return proba, logits, embeddings


class OdorGrapher(TorchModel):
    def __init__(self,
                 labels: int,
                 class_imbalance_ratio: Optional[List] = None,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 batch_size: int = 128,
                 node_out_feats: int = 128,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 4,
                 number_atom_features: int = 147,
                 number_bond_features: int = 7,
                 readout: str = 'set2set',
                 num_step_set2set: int = 6,
                 num_layer_set2set: int = 3,
                 ffn_hidden_list: List = [300],
                 ffn_output_dim: int = 256,
                 ffn_activation: str = 'relu',
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True,
                 weight_decay: float = 1e-5,
                 self_loop: bool = False,
                 optimizer: str = 'adam',
                 device_name: Optional[str] = None,
                 gate_temp: float = 1.2,
                 num_heads: int = 6,
                 gap_weight: float = 0.0,
                 gap_margin: float = 0.3,
                 gap_temp_neg: float = 0.5,
                 tau: float = 0.5,
                 **kwargs):
        model: nn.Module = OdorGrapher_NN(
            labels=labels,
            node_out_feats=node_out_feats,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
            number_atom_features=number_atom_features,
            number_bond_features=number_bond_features,
            readout=readout,
            num_step_set2set=num_step_set2set,
            num_layer_set2set=num_layer_set2set,
            ffn_hidden_list=ffn_hidden_list,
            ffn_output_dim=ffn_output_dim,
            ffn_activation=ffn_activation,
            ffn_dropout_p=ffn_dropout_p,
            ffn_dropout_at_input_no_act=ffn_dropout_at_input_no_act,
            gate_temp=gate_temp,
            num_heads=num_heads,

        )

        if class_imbalance_ratio is not None:
            ratio_tensor = torch.tensor(class_imbalance_ratio, dtype=torch.float32)
            self.rare_label_indices = [
                i for i, r in enumerate(ratio_tensor.tolist()) if r > 3
            ]
        else:
            self.rare_label_indices = []
    
        self.contrastive_margin = 0.3         
        self.alpha_contrastive = 1.0
      
        loss = AdaptiveLogitAdjustedASL(
            class_imbalance_ratio=class_imbalance_ratio,
            tau=tau, gamma_pos=0.0,
            gamma_neg_head=4.0, 
            gamma_neg_tail=2.0,
            ir_threshold=3.0, 
            clip=0.05, 
            beta_pos=0.6,
            reduction="sum",
            gap_weight=gap_weight, 
            gap_margin=gap_margin, 
            gap_temp_neg=gap_temp_neg
        )
        output_types = ['prediction', 'loss', 'embedding']

        opt_name = optimizer.lower()
        opt_map = {
            "adam": Adam,
            "adagrad": AdaGrad,
            "adamw": AdamW,
            "sparseadam": SparseAdam,
            "rmsprop": RMSProp,
            "sgd": GradientDescent
        }
        
        optimizer_cls = opt_map.get(opt_name)
        if optimizer_cls is None:
            print(f"[Warning] Unsupported optimizer '{optimizer}', falling back to Adam.")
            optimizer = Adam()
        else:
            optimizer = optimizer_cls()
            
        optimizer.learning_rate = learning_rate
        if device_name is not None:
            device: Optional[torch.device] = torch.device(device_name)
        else:
            device = None
        super(OdorGrapher, self).__init__(model,
                                           loss=loss,
                                           output_types=output_types,
                                           optimizer=optimizer,
                                           learning_rate=learning_rate,
                                           batch_size=batch_size,
                                           device=device,
                                           **kwargs)

        self.weight_decay: float = weight_decay
        self._self_loop: bool = self_loop
        self.regularization_loss: Callable = self._regularization_loss

    def _regularization_loss(self) -> torch.Tensor:
        l1_regularization: torch.Tensor = torch.tensor(0., requires_grad=True)
        l2_regularization: torch.Tensor = torch.tensor(0., requires_grad=True)
        for name, param in self.model.named_parameters():
            if 'bias' not in name:
                l1_regularization = l1_regularization + torch.norm(param, p=1)
                l2_regularization = l2_regularization + torch.norm(param, p=2)
        l1_norm: torch.Tensor = self.weight_decay * l1_regularization
        l2_norm: torch.Tensor = self.weight_decay * l2_regularization
        return l1_norm + l2_norm

    def _prepare_batch(
        self, batch: Tuple[List, List, List]
    ) -> Tuple[DGLGraph, List[torch.Tensor], List[torch.Tensor]]:
        inputs: List
        labels: List
        weights: List

        inputs, labels, weights = batch
        dgl_graphs: List[DGLGraph] = [
            graph.to_dgl_graph(self_loop=self._self_loop)
            for graph in inputs[0]
        ]

        g: DGLGraph = dgl.batch(dgl_graphs).to(self.device)
        _, labels, weights = super(OdorGrapher, self)._prepare_batch(
            ([], labels, weights))

        return g, labels, weights