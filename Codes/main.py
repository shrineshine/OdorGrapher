import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import tempfile
from typing import Dict, List, Tuple
import numpy as np
import torch
import deepchem as dc
from tqdm import tqdm
import utils
import evaluate
from models import newModel
from config import *

class GraphConstants:
    MAX_ATOMIC_NUM = 100
    ATOM_FEATURES = {'valence': [0, 1, 2, 3, 4, 5, 6], 'degree': [0, 1, 2, 3, 4, 5], 'num_Hs': [0, 1, 2, 3, 4], 'formal_charge': [-1, -2, 1, 2, 3, 0], 'atomic_num': list(range(MAX_ATOMIC_NUM)), 'hybridization': ['SP', 'SP2', 'SP3']}
    BOND_FDIM = 7
SMARTS_GROUPS = {'carbonyl': '[CX3]=[OX1]', 'aldehyde': '[CX3H1](=O)[#6]', 'ketone': '[CX3](=O)[#6]', 'carboxylic_acid': 'C(=O)[OH]', 'ester': '[CX3](=O)[OX2H0][#6]', 'ether': '[OD2]([#6])[#6]', 'amine_primary': '[NX3;H2;!$(NC=O)]', 'amine_secondary': '[NX3;H1;!$(NC=O)]', 'thiol': '[SX2H]', 'thioether': '[#16X2][#6]', 'aromatic_ring': 'a1aaaaa1', 'phenol': 'c1ccc(cc1)[OH]', 'alkene': 'C=C', 'alkyne': 'C#C', 'nitro': '[$([NX3](=O)=O)]'}

def set_all_seeds(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import dgl
        if hasattr(dgl, 'seed'):
            dgl.seed(seed)
    except Exception:
        pass

def get_feature_dims(use_fg: bool):
    base_dim = sum((len(choices) for choices in GraphConstants.ATOM_FEATURES.values())) + 5
    if use_fg:
        base_dim += len(SMARTS_GROUPS)
    return (base_dim, GraphConstants.BOND_FDIM)

def build_label_family_prior(train_dataset, n_families: int=12, temp: float=0.75):
    y = np.asarray(train_dataset.y)
    w = np.asarray(train_dataset.w if train_dataset.w is not None else np.ones_like(y))
    pos = ((y == 1) & (w != 0)).astype(np.float32)
    n_samples, n_tasks = pos.shape
    counts = pos.sum(axis=0).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    cooc = pos.T @ pos
    cond = cooc / counts[:, None]
    prev = counts / max(float(n_samples), 1.0)
    sym = 0.5 * (cond + cond.T)
    lift = sym / np.clip((prev[:, None] + prev[None, :]) * 0.5, 1e-06, None)
    sim = sym * np.log1p(lift)
    np.fill_diagonal(sim, sim.diagonal() + 1.0)
    k = max(2, min(int(n_families), n_tasks))
    anchors = [int(np.argmax(counts))]
    for _ in range(1, k):
        max_sim = sim[:, anchors].max(axis=1)
        diversity = 1.0 - max_sim / (max_sim.max() + 1e-08)
        score = counts * diversity
        score[anchors] = -1.0
        anchors.append(int(np.argmax(score)))
    logits = sim[:, anchors] / max(float(temp), 1e-06)
    logits = logits - logits.max(axis=1, keepdims=True)
    prior = np.exp(logits)
    prior = prior / np.clip(prior.sum(axis=1, keepdims=True), 1e-12, None)
    return prior.astype(np.float32)

def build_label_competitor_prior(train_dataset, topk: int=6, min_cond: float=0.2, freq_ratio: float=1.3, temp: float=0.7, ir_threshold: float=3.0):
    y = np.asarray(train_dataset.y)
    w = np.asarray(train_dataset.w if train_dataset.w is not None else np.ones_like(y))
    pos = ((y == 1) & (w != 0)).astype(np.float32)
    n_samples, n_tasks = pos.shape
    counts = pos.sum(axis=0).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    cooc = pos.T @ pos
    cond = cooc / counts[:, None]
    max_count = float(counts.max())
    imbalance = max_count / counts
    prior = np.zeros((n_tasks, n_tasks), dtype=np.float32)
    for i in range(n_tasks):
        if imbalance[i] < float(ir_threshold):
            continue
        score = np.zeros(n_tasks, dtype=np.float32)
        for j in range(n_tasks):
            if i == j:
                continue
            if counts[j] <= counts[i] * float(freq_ratio):
                continue
            cij = float(cond[i, j])
            if cij < float(min_cond):
                continue
            score[j] = cij * np.log1p(counts[j] / counts[i])
        nz = np.where(score > 0)[0]
        if nz.size == 0:
            continue
        order = nz[np.argsort(-score[nz])]
        keep = order[:max(1, int(topk))]
        logits = score[keep] / max(float(temp), 1e-06)
        logits = logits - logits.max()
        weights = np.exp(logits)
        weights = weights / np.clip(weights.sum(), 1e-12, None)
        prior[i, keep] = weights.astype(np.float32)
    return prior

def build_label_cardinality_stats(train_dataset):
    y = np.asarray(train_dataset.y)
    w = np.asarray(train_dataset.w if train_dataset.w is not None else np.ones_like(y))
    pos = ((y == 1) & (w != 0)).astype(np.float32)
    counts = pos.sum(axis=1).astype(np.float32)
    counts = counts[counts > 0]
    if counts.size == 0:
        return (4.0, 1.5)
    mean_count = float(counts.mean())
    std_count = float(counts.std())
    return (mean_count, max(1.0, std_count))

def build_label_hub_score(train_dataset, ir_threshold: float=3.0):
    y = np.asarray(train_dataset.y)
    w = np.asarray(train_dataset.w if train_dataset.w is not None else np.ones_like(y))
    pos = ((y == 1) & (w != 0)).astype(np.float32)
    n_samples, n_tasks = pos.shape
    counts = pos.sum(axis=0).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    cooc = pos.T @ pos
    cond = cooc / counts[:, None]
    max_count = float(counts.max())
    imbalance = max_count / counts
    tail_w = 1.0 / (1.0 + np.exp(-2.0 * (np.log(imbalance) - np.log(float(ir_threshold)))))
    score = np.zeros(n_tasks, dtype=np.float32)
    for j in range(n_tasks):
        lift_sum = 0.0
        mass_sum = 0.0
        for i in range(n_tasks):
            if i == j or counts[j] <= counts[i]:
                continue
            cij = float(cond[i, j])
            if cij <= 0.0:
                continue
            rarity_gap = np.log1p(counts[j] / counts[i])
            w_ij = float(tail_w[i]) * rarity_gap
            lift_sum += w_ij * cij
            mass_sum += w_ij
        if mass_sum > 0:
            score[j] = lift_sum / mass_sum
    if float(score.max()) > 0:
        score = score / float(score.max())
    prev = counts / counts.max()
    score = 0.65 * score + 0.35 * prev
    score = np.clip(score, 0.0, 1.0)
    return score.astype(np.float32)

def score_model(model, dataset) -> Dict[str, float]:
    scores = evaluate.evaluate_model(model, dataset)
    return {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in scores.items()}

def state_dict_to_cpu(model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model_state.items()}

def average_state_dicts(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(states) == 1:
        return state_dict_to_cpu(states[0])
    avg = {}
    keys = states[0].keys()
    for k in keys:
        ref = states[0][k]
        if torch.is_floating_point(ref):
            stacked = torch.stack([s[k].float() for s in states], dim=0)
            avg[k] = stacked.mean(dim=0).to(dtype=ref.dtype)
        else:
            avg[k] = ref.clone()
    return avg

def build_model(train_dataset, model_dir_thisfold: str, learning_rate, label_family_prior, label_competitor_prior, label_hub_score, toppush_weight_this):
    n_tasks = len(train_dataset.tasks)
    card_center, card_scale = build_label_cardinality_stats(train_dataset)
    train_ratios = utils.get_class_imbalance_ratio(train_dataset)
    atom_fdim, bond_fdim = get_feature_dims(use_fg)
    model = newModel(n_tasks=n_tasks, batch_size=batch_size, learning_rate=learning_rate, class_imbalance_ratio=train_ratios, label_family_prior=label_family_prior, loss_aggr_type='sum', node_out_feats=node_out_feats, edge_hidden_feats=128, edge_out_feats=128, num_step_message_passing=num_mp, mpnn_residual=True, message_aggregator_type='sum', mode='classification', number_atom_features=atom_fdim, number_bond_features=bond_fdim, n_classes=1, readout_type='attn_pool', num_step_set2set=3, num_layer_set2set=2, ffn_hidden_list=[ffn_hidden, 256], ffn_embeddings=256, ffn_activation='relu', ffn_dropout_p=dropout, ffn_dropout_at_input_no_act=False, weight_decay=1e-05, self_loop=False, optimizer_name='adam', log_frequency=32, model_dir=model_dir_thisfold, device_name='cuda:0', tau=tau, toppush_weight=toppush_weight_this, toppush_margin=toppush_margin, toppush_temp=toppush_temp, toppush_tail_scale=toppush_tail_scale, family_prompt_dim=family_prompt_dim, family_prompt_heads=family_prompt_heads, family_adapter_dropout=family_adapter_dropout, family_adapter_rank=family_adapter_rank, family_delta_scale_init=family_delta_scale_init, family_bias_scale_init=family_bias_scale_init, label_competitor_prior=label_competitor_prior, label_hub_score=label_hub_score, family_competitor_delta_gain=family_competitor_delta_gain, family_competitor_bias_suppress=family_competitor_bias_suppress, family_competitor_floor=family_competitor_floor, family_diffuse_tail_gain=family_diffuse_tail_gain, family_reliability_min=family_reliability_min, label_cardinality_center=card_center, label_cardinality_scale=card_scale, family_cardinality_blend=family_cardinality_blend, family_cardinality_min_gate=family_cardinality_min_gate, family_hub_delta_gain=family_hub_delta_gain, family_hub_bias_suppress=family_hub_bias_suppress, family_ultra_tail_threshold=family_ultra_tail_threshold, family_hub_competitor_mix=family_hub_competitor_mix, family_hub_self_margin=family_hub_self_margin)
    return model

def main():
    dataset_kfold_root = '../train_data/dataset/frethres=30_fg_off/cvdata'
    all_val_scores = []
    with tempfile.TemporaryDirectory(prefix='odorgrapher_') as temp_root:
        for fold_idx in range(1, k + 1):
            print(f'\n===== Fold {fold_idx} / {k} =====')
            set_all_seeds(42)
            train_dataset = dc.data.DiskDataset(os.path.join(dataset_kfold_root, f'fold_{fold_idx}', 'train_data'))
            val_dataset = dc.data.DiskDataset(os.path.join(dataset_kfold_root, f'fold_{fold_idx}', 'cv_data'))
            print('train_dataset:', len(train_dataset))
            print('val_dataset:', len(val_dataset))
            family_prior = build_label_family_prior(train_dataset, n_families=family_n_prompts, temp=family_prior_temp)
            competitor_prior = build_label_competitor_prior(train_dataset, topk=family_competitor_topk, min_cond=family_competitor_min_cond, freq_ratio=family_competitor_freq_ratio, temp=family_competitor_temp, ir_threshold=3.0)
            hub_score = build_label_hub_score(train_dataset, ir_threshold=3.0)
            learning_rate = dc.models.optimizers.ExponentialDecay(initial_rate=lr, decay_rate=0.5, decay_steps=len(train_dataset) * 20 / batch_size, staircase=True)
            model_dir_thisfold = os.path.join(temp_root, f'fold_{fold_idx}')
            os.makedirs(model_dir_thisfold, exist_ok=True)
            model = build_model(train_dataset=train_dataset, model_dir_thisfold=model_dir_thisfold, learning_rate=learning_rate, label_family_prior=family_prior, label_competitor_prior=competitor_prior, label_hub_score=hub_score, toppush_weight_this=toppush_weight)
            best_val_pr_auc = -float('inf')
            best_epoch = -1
            epochs_no_improve = 0
            best_single_state = None
            soup_candidates: List[Tuple[float, int, Dict[str, torch.Tensor]]] = []
            for epoch in tqdm(range(1, nb_epoch + 1)):
                loss = model.fit(train_dataset, nb_epoch=1, max_checkpoints_to_keep=0, deterministic=True, restore=False)
                val_scores = score_model(model, val_dataset)
                val_macro_pr_auc = val_scores.get('PR-AUC(macro)', 0.0)
                print(f'Epoch {epoch}/{nb_epoch} | Loss: {loss:.4f} | Valid PR-AUC(macro): {val_macro_pr_auc:.4f}')
                state_cpu = state_dict_to_cpu(model.model.state_dict())
                if len(soup_candidates) < soup_topk or val_macro_pr_auc > soup_candidates[-1][0]:
                    soup_candidates.append((float(val_macro_pr_auc), int(epoch), state_cpu))
                    soup_candidates = sorted(soup_candidates, key=lambda x: (-x[0], x[1]))[:soup_topk]
                if val_macro_pr_auc > best_val_pr_auc:
                    best_val_pr_auc = val_macro_pr_auc
                    best_epoch = epoch
                    epochs_no_improve = 0
                    best_single_state = state_cpu
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break
            if best_single_state is None:
                raise RuntimeError(f'Fold {fold_idx} did not produce a valid model state.')
            model.model.load_state_dict(best_single_state, strict=True)
            single_scores = score_model(model, val_dataset)
            best_stage1_state = best_single_state
            final_mode = 'single'
            if len(soup_candidates) >= 2:
                soup_epochs = [x[1] for x in soup_candidates]
                avg_state = average_state_dicts([x[2] for x in soup_candidates])
                model.model.load_state_dict(avg_state, strict=True)
                soup_scores = score_model(model, val_dataset)
                if soup_scores.get('PR-AUC(macro)', -1000000000.0) >= single_scores.get('PR-AUC(macro)', -1000000000.0):
                    best_stage1_state = avg_state
                    final_mode = f'soup@{soup_epochs}'
                else:
                    model.model.load_state_dict(best_single_state, strict=True)
            model.model.load_state_dict(best_stage1_state, strict=True)
            score_model(model, val_dataset)
            final_scores = score_model(model, val_dataset)
            final_mode = f'stage1_{final_mode}'
            final_scores = {name: round(value, 4) if isinstance(value, (int, float, np.floating)) else value for name, value in final_scores.items()}
            all_val_scores.append(final_scores)
            print(f'Fold {fold_idx} | Best Epoch = {best_epoch} | Mode = {final_mode} | Val Scores: {final_scores}')
    metrics = list(all_val_scores[0].keys())
    mean_scores = {metric: np.mean([fold_scores[metric] for fold_scores in all_val_scores]) for metric in metrics}
    std_scores = {metric: np.std([fold_scores[metric] for fold_scores in all_val_scores]) for metric in metrics}
    print('\n===== Final Mean ± Std over folds =====')
    for metric in metrics:
        print(f'{metric}: {mean_scores[metric]:.4f} ± {std_scores[metric]:.4f}')
if __name__ == '__main__':
    main()
