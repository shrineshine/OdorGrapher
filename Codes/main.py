import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import tempfile
import pandas as pd
import numpy as np
import torch
import deepchem as dc
import utils
import evaluate 
from tqdm import tqdm
from typing import List, Tuple, Optional
from deepchem.data.datasets import DiskDataset
from skmultilearn.model_selection import IterativeStratification
from graph_featurizer import GraphConvConstants
from odorgrapher import OdorGrapher

def load_if_exists(path: str) -> Optional[DiskDataset]:
    try:
        meta1 = os.path.join(path, "metadata.csv.gzip")
        if os.path.exists(meta1):
            return DiskDataset(path)
    except:
        pass
    return None

def iterative_stratified_kfold_dc(
    dataset: DiskDataset,
    k: int = 5,
    order: int = 2,
    directories: Optional[List[str]] = None
) -> List[Tuple[DiskDataset, DiskDataset]]:

    assert k > 1

    if directories is None:
        directories = [tempfile.mkdtemp() for _ in range(2 * k)]
    else:
        assert len(directories) == 2 * k

    X = pd.DataFrame(dataset.X)
    y = pd.DataFrame(dataset.y)

    stratifier = IterativeStratification(
        n_splits=k,
        order=order
    )
    split_gen = stratifier.split(X, y)

    folds = []

    for fold in range(k):
        train_dir = directories[2 * fold]
        cv_dir = directories[2 * fold + 1]

        train_exists = load_if_exists(train_dir)
        cv_exists = load_if_exists(cv_dir)

        if train_exists is not None and cv_exists is not None:
            print(f"[Skip] Fold {fold+1} already exists. Loading saved datasets.")
            folds.append((train_exists, cv_exists))
            continue

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(cv_dir, exist_ok=True)

        train_idx, cv_idx = next(split_gen)

        train_dataset = dataset.select(train_idx.tolist(), train_dir)
        cv_dataset = dataset.select(cv_idx.tolist(), cv_dir)

        folds.append((train_dataset, cv_dataset))
        print(f"[Generate] Fold {fold+1} generated and saved.")

    return folds

K = 5
thres = 30
dataset_dir = "../data/dataset"
results_dir = "../results"
dataset_path = os.path.join(dataset_dir, f"frethres={thres}")
dataset = dc.data.DiskDataset(dataset_path)
print("Loaded dataset size:", len(dataset))

directories = [""] * (2 * K)
for fold in range(K):
    directories[2 * fold] = f"../data/cvdata/fold_{fold+1}/train_data"
    directories[2 * fold + 1] = f"../data/cvdata/fold_{fold+1}/cv_data"

folds = iterative_stratified_kfold_dc(
    dataset=dataset,
    k=K,
    order=2,
    directories=directories
)

print("Total folds generated:", len(folds))
for i, (train_ds, cv_ds) in enumerate(folds):
    print(f"Fold {i+1}: train={len(train_ds)}, cv={len(cv_ds)}")


def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

k = 5
nb_epoch = 1000
patience = 50
batch_size = 512
lr = 1e-4
dropout = 0.4

os.makedirs(results_dir, exist_ok=True)
dataset_kfold_root = "../data/cvdata"

all_val_scores = []
all_history_loss = []
all_history_valid_pr = []

for fold_idx in range(1, k+1):
    print(f"\n===== Fold {fold_idx} / {k} =====")
    set_all_seeds(42)
    train_dataset = dc.data.DiskDataset(os.path.join(dataset_kfold_root, f"fold_{fold_idx}", "train_data"))
    val_dataset  = dc.data.DiskDataset(os.path.join(dataset_kfold_root, f"fold_{fold_idx}", "cv_data"))
    print("train_dataset: ", len(train_dataset))
    print("val_dataset: ", len(val_dataset))

    labels = len(train_dataset.tasks)
    print("labels:", labels)

    train_ratios = utils.get_class_imbalance_ratio(train_dataset)
    learning_rate = dc.models.optimizers.ExponentialDecay(
        initial_rate=lr,
        decay_rate=0.5,
        decay_steps=len(train_dataset) * 10 / batch_size,
        staircase=True
    )

    model_dir_thisfold = os.path.join(results_dir, f'fold_{fold_idx}')
    os.makedirs(model_dir_thisfold, exist_ok=True)
    model = OdorGrapher(labels = labels,
                        batch_size = batch_size,
                        learning_rate = learning_rate,
                        class_imbalance_ratio = train_ratios,
                        node_out_feats = 128,
                        edge_hidden_feats = 128,
                        num_step_message_passing = 4,
                        number_atom_features = GraphConvConstants.ATOM_FDIM,
                        number_bond_features = GraphConvConstants.BOND_FDIM,
                        readout = 'attn_pool',
                        num_step_set2set = 2,
                        num_layer_set2set = 2,
                        ffn_hidden_list= [640, 256],
                        ffn_output_dim = 256,
                        ffn_activation = 'relu',
                        ffn_dropout_p = dropout,
                        ffn_dropout_at_input_no_act = False,
                        weight_decay = 1e-5,
                        self_loop = False,
                        optimizer = 'adam',
                        log_frequency = 32,
                        model_dir = model_dir_thisfold,
                        device_name='cuda:0',
                        tau=0.7
    )
    best_val_pr_auc = -float('inf')
    best_epoch = -1
    epochs_no_improve = 0

    history_epochs = []
    history_loss = []
    history_valid_pr = []

    for epoch in tqdm(range(1, nb_epoch + 1)):
        loss = model.fit(
            train_dataset,
            nb_epoch=1,
            max_checkpoints_to_keep=0,
            deterministic=True,
            restore=False
        )
        val_scores = evaluate.evaluate_model(model, val_dataset)
        val_macro_pr_auc = val_scores.get("PR-AUC(macro)", 0.0)
        print(
            f"Epoch {epoch}/{nb_epoch} | Loss: {loss:.4f} | "
            f"Valid PR-AUC(macro): {val_macro_pr_auc:.4f}"
        )
        history_epochs.append(epoch)
        history_loss.append(float(loss))
        history_valid_pr.append(float(val_macro_pr_auc))
        
        if val_macro_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_macro_pr_auc
            best_epoch = epoch
            epochs_no_improve = 0
            model.save_checkpoint(max_checkpoints_to_keep=1)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    all_history_loss.append(history_loss)
    all_history_valid_pr.append(history_valid_pr)

    model.restore(os.path.join(model_dir_thisfold, "checkpoint1.pt"))
    best_val_scores = evaluate.evaluate_model(model, val_dataset)
    best_val_scores = {k: round(v, 4) if isinstance(v, (int, float)) else v
                   for k, v in best_val_scores.items()}
    all_val_scores.append(best_val_scores)

    print(f"Fold {fold_idx} | Best Epoch = {best_epoch} | Val Scores: {best_val_scores}")