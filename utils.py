import pandas as pd
import torch
import numpy as np
from typing import List, Optional
from deepchem.data.datasets import DiskDataset, NumpyDataset

def get_class_imbalance_ratio(dataset: DiskDataset, log_transform: bool = True, clip_max: float = 100.0) -> List[float]:
    if not isinstance(dataset, (DiskDataset, NumpyDataset)):
        raise Exception("The dataset should be a deepchem DiskDataset or NumpyDataset")

    y = dataset.y 
    df = pd.DataFrame(y)
    
    pos_counts = (df == 1).sum().to_numpy() + 1e-6
    neg_counts = (df == 0).sum().to_numpy() + 1e-6
    
    ratio = neg_counts / pos_counts  

    if log_transform:
        ratio = np.log1p(ratio) 
    
    ratio = np.clip(ratio, a_min=1e-2, a_max=clip_max) 
    
    return ratio.tolist()

def split_labels_head_medium_tail(dataset, head_label="sweet"):
    from openpom.utils.data_utils import get_class_imbalance_ratio
    
    ratios = get_class_imbalance_ratio(dataset, log_transform=False)  
    label_names = dataset.tasks
    
    y = dataset.y
    pos_counts = (y == 1).sum(axis=0)
    
    label_info = [(name, pos_counts[idx]) for idx, name in enumerate(label_names) if name != head_label]
    label_info.sort(key=lambda x: x[1], reverse=True)  

    split_point = len(label_info) // 2
    medium_labels = [name for name, _ in label_info[:split_point]]
    tail_labels = [name for name, _ in label_info[split_point:]]
    
    return [head_label], medium_labels, tail_labels

def load_label_graph(similarity_csv: str, topk: int = 5, min_sim: float = 0.8) -> torch.Tensor:
    sim_df = pd.read_csv(similarity_csv, index_col=0)
    sim_tensor = torch.tensor(sim_df.values, dtype=torch.float32)
    sim_tensor.fill_diagonal_(0.0)

    n = sim_tensor.shape[0]
    adj = torch.zeros_like(sim_tensor)

    for i in range(n):
        topk_idx = torch.topk(sim_tensor[i], k=topk).indices

        added = False
        for j in topk_idx:
            if sim_tensor[i, j] >= min_sim:
                adj[i, j] = sim_tensor[i, j]
                added = True

        if not added:
            j = topk_idx[0]
            adj[i, j] = sim_tensor[i, j]

    adj = (adj + adj.T) / 2
    return adj

def get_class_imbalance_ratio_v2(
    dataset: DiskDataset,
    log_transform: bool = False,          
    clip_max: Optional[float] = None,    
    alpha: float = 0.5,                  
    use_sample_weights: bool = True       
) -> List[float]:
    if not isinstance(dataset, (DiskDataset, NumpyDataset)):
        raise Exception("The dataset should be a deepchem DiskDataset or NumpyDataset")

    y = np.asarray(dataset.y)                      
    w = np.asarray(dataset.w) if (use_sample_weights and hasattr(dataset, "w")) else np.ones_like(y, dtype=float)

    valid = ((y == 0) | (y == 1)) & (w > 0)

    T = y.shape[1]
    ir = np.zeros(T, dtype=np.float64)

    for t in range(T):
        mask_t = valid[:, t]
        if not np.any(mask_t):
            ir[t] = 1.0
            continue
        y_t = y[mask_t, t]
        w_t = w[mask_t, t]

        pos = np.sum(w_t * (y_t == 1))
        neg = np.sum(w_t * (y_t == 0))

        pos = pos + alpha
        neg = neg + alpha

        ir[t] = neg / max(pos, 1e-12)

    if clip_max is not None:
        ir = np.clip(ir, a_min=1e-6, a_max=float(clip_max))
    else:
        ir = np.clip(ir, a_min=1e-6, a_max=1e12)

    if log_transform:
        ir = np.log1p(ir)

    return ir.tolist()