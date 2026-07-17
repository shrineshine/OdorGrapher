import os
import re
import ast
import json
import numpy as np
import pandas as pd
import pyrfume
import deepchem as dc
from tqdm.auto import tqdm
from collections import Counter
from graph_featurizer import GraphFeaturizer
from skmultilearn.model_selection import IterativeStratification
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
tqdm.pandas()

PREFIX = "/data/pyrfume_data"
RAW_PATH = "../train_data/raw"
DATASET_ROOT = "../train_data/dataset"

THRESHOLDS = [10, 20, 30, 50]
K = 5

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(DATASET_ROOT, exist_ok=True)

FULL_ROOT = os.path.join(DATASET_ROOT, "FULL_MINTHRES")  
os.makedirs(FULL_ROOT, exist_ok=True)

def safe_load(path):
    full = os.path.join(PREFIX, path)
    if not os.path.exists(full):
        raise FileNotFoundError(full)
    return pyrfume.load_data(full, remote=False).reset_index()

def load_or_build_raw():
    goodscents_path = os.path.join(RAW_PATH, "goodscents_merged.csv")
    leffingwell_path = os.path.join(RAW_PATH, "leffingwell_merged.csv")
    flavordb_path = os.path.join(RAW_PATH, "flavordb_merged.csv")

    if not os.path.exists(goodscents_path):
        behavior = safe_load("goodscents/behavior.csv")
        stimuli = safe_load("goodscents/stimuli.csv")
        molecules = safe_load("goodscents/molecules.csv")
        merged = pd.merge(stimuli, behavior, on="Stimulus", how="outer")
        merged = pd.merge(merged, molecules, on="CID", how="outer")
        merged.to_csv(goodscents_path, index=False)

    if not os.path.exists(leffingwell_path):
        behavior = safe_load("leffingwell/behavior_sparse.csv")
        stimuli = safe_load("leffingwell/stimuli.csv")
        merged = pd.merge(stimuli, behavior, on="Stimulus", how="outer")
        merged.to_csv(leffingwell_path, index=False)

    if not os.path.exists(flavordb_path):
        behavior = safe_load("flavordb/behavior.csv")
        stimuli = safe_load("flavordb/stimuli.csv")
        molecules = safe_load("flavordb/molecules.csv")
        merged = pd.merge(stimuli, behavior, on="Stimulus", how="outer")
        merged = pd.merge(merged, molecules, on="CID", how="outer")
        merged.to_csv(flavordb_path, index=False)

    return (
        pd.read_csv(goodscents_path),
        pd.read_csv(leffingwell_path),
        pd.read_csv(flavordb_path),
    )

STOPWORDS = {"of","the","de","and","or","new","almost","very","more","less","with","without"}
NON_ODOR_WORDS = {"strong","weak","pleasant","unpleasant","characteristic"}

def split_clean(s: str):
    return [re.sub(r"\s+", " ", x.strip().lower()) for x in s.split(";") if x.strip()]

def extract_tokens(desc: str):
    return [t for t in desc.split() if t.isalpha() and t not in STOPWORDS and t not in NON_ODOR_WORDS]

def build_global_counter(gs_df, lf_df, fldb_df) -> Counter:
    counter = Counter()
    sources = [
        (gs_df, "Descriptors"),
        (lf_df, "Labels"),
        (fldb_df, "Odor Percepts")
    ]
    for df, col in sources:
        if col not in df.columns:
            continue
        for i in tqdm(df[col], desc=f"Counting tokens from {col}", leave=False):
            if isinstance(i, str):
                items = ast.literal_eval(i) if col == "Labels" else split_clean(i)
                for raw in items:
                    for t in extract_tokens(raw):
                        counter[t] += 1
    return counter

def make_merged_df(gs_df, lf_df, fldb_df) -> pd.DataFrame:
    gs = gs_df.rename(columns={"IsomericSMILES": "SMILES_gs"}).copy()
    lf = lf_df.rename(columns={"IsomericSMILES": "SMILES_lf"}).copy()
    fd = fldb_df.rename(columns={"IsomericSMILES": "SMILES_fldb"}).copy()

    merged = gs.merge(lf, on="CID", how="outer")
    merged = merged.merge(fd, on="CID", how="outer")

    def resolve_smiles(row):
        for col in ["SMILES_gs", "SMILES_lf", "SMILES_fldb"]:
            v = row.get(col)
            if isinstance(v, str) and len(v) > 0:
                return v
        return None

    merged["IsomericSMILES"] = merged.apply(resolve_smiles, axis=1)
    merged = merged.dropna(subset=["IsomericSMILES"])
    return merged

def build_full_once(merged_df: pd.DataFrame,
                    global_counter: Counter,
                    min_threshold: int) -> tuple[dc.data.DiskDataset, list[str], str]:
    full_dir = os.path.join(FULL_ROOT, f"minthres={min_threshold}")
    os.makedirs(full_dir, exist_ok=True)

    dataset_dir = os.path.join(full_dir, "full_dataset")
    tasks_path = os.path.join(full_dir, "full_tasks.json")
    vector_csv = os.path.join(full_dir, "vector_full.csv")

    if os.path.exists(dataset_dir) and os.path.exists(tasks_path):
        with open(tasks_path, "r") as f:
            full_tasks = json.load(f)
        print(f"[FULL] Reusing existing full dataset: {dataset_dir}")
        return dc.data.DiskDataset(dataset_dir), full_tasks, full_dir

    full_counter = Counter({k_: v for k_, v in global_counter.items() if v >= min_threshold})
    full_tasks = sorted(full_counter.keys())
    if len(full_tasks) == 0:
        raise RuntimeError(f"[FULL] No tasks found with min_threshold={min_threshold}")

    print(f"[FULL] Building full tasks with min_threshold={min_threshold}, tasks={len(full_tasks)}")

    def build_label_row(row):
        d = dict.fromkeys(full_tasks, 0)
        for col in ["Descriptors", "Labels", "Odor Percepts"]:
            v = row.get(col)
            if isinstance(v, str):
                items = ast.literal_eval(v) if col == "Labels" else split_clean(v)
                for raw in items:
                    for t in extract_tokens(raw):
                        if t in d:
                            d[t] = 1
        return d

    print("[FULL] Building label matrix (one-time)...")
    label_df = merged_df.progress_apply(build_label_row, axis=1, result_type="expand")
    vector_df = pd.concat([merged_df[["IsomericSMILES"]], label_df], axis=1)
    vector_df.to_csv(vector_csv, index=False)

    with open(tasks_path, "w") as f:
        json.dump(full_tasks, f, ensure_ascii=False, indent=2)

    print("[FULL] Featurizing molecules (one-time, this is the heavy step)...")
    featurizer = GraphFeaturizer()
    loader = dc.data.CSVLoader(tasks=full_tasks,
                              feature_field="IsomericSMILES",
                              featurizer=featurizer)
    full_dataset = loader.create_dataset([vector_csv])

    if len(full_dataset) == 0:
        raise RuntimeError("[FULL] Empty dataset after featurization")

    os.makedirs(dataset_dir, exist_ok=True)
    full_dataset.move(dataset_dir)

    print(f"[FULL] Saved full dataset to: {dataset_dir}")
    return dc.data.DiskDataset(dataset_dir), full_tasks, full_dir

def slice_dataset_for_threshold(full_ds: dc.data.DiskDataset,
                               full_tasks: list[str],
                               global_counter: Counter,
                               min_freq: int,
                               k: int,
                               use_fg: bool = True) -> tuple[dc.data.DiskDataset, list[str], str]:
    suffix = "fg_on" if use_fg else "fg_off"
    out_dir = os.path.join(DATASET_ROOT, f"frethres={min_freq}_{suffix}") # 唯一路径
    os.makedirs(out_dir, exist_ok=True)

    out_dataset_dir = os.path.join(out_dir, "dataset")
    out_tasks_path = os.path.join(out_dir, "tasks.json")

    if os.path.exists(out_dataset_dir) and os.path.exists(out_tasks_path):
        with open(out_tasks_path, "r") as f:
            tasks = json.load(f)
        print(f"[{suffix}-th={min_freq}] Reusing existing sliced dataset: {out_dataset_dir}")
        return dc.data.DiskDataset(out_dataset_dir), tasks, out_dir

    candidate = sorted([t for t, c in global_counter.items() if c >= min_freq])
    idx_map = {t: i for i, t in enumerate(full_tasks)}
    
    y_full = full_ds.y
    valid_idx = []
    valid_tasks = []
    
    for t in candidate:
        if t not in idx_map: continue
        i = idx_map[t]
        pos = int(np.sum(y_full[:, i] == 1))
        if pos >= min_freq and pos >= k:
            valid_idx.append(i)
            valid_tasks.append(t)

    print(f"[{suffix}-th={min_freq}] valid tasks={len(valid_tasks)}")

    if len(valid_idx) == 0:
        raise RuntimeError(f"[{suffix}-th={min_freq}] No valid tasks found")

    y_new = y_full[:, valid_idx]
    row_sum = np.sum(y_new == 1, axis=1)
    keep_rows = row_sum > 0
    
    X_new = full_ds.X[keep_rows]
    y_new = y_new[keep_rows]
    ids_new = full_ds.ids[keep_rows]
    
    if full_ds.w is not None and full_ds.w.size:
        w_new = full_ds.w[keep_rows][:, valid_idx]
    else:
        w_new = np.ones_like(y_new)
    
    dc.data.DiskDataset.from_numpy(
        X=X_new, y=y_new, w=w_new, ids=ids_new,
        data_dir=out_dataset_dir
    )

    with open(out_tasks_path, "w") as f:
        json.dump(valid_tasks, f, ensure_ascii=False, indent=2)

    return dc.data.DiskDataset(out_dataset_dir), valid_tasks, out_dir

def iterative_stratified_kfold_save(dataset: dc.data.DiskDataset, k: int, save_root: str):
    X = pd.DataFrame(dataset.X)  # skmultilearn 需要 array-like
    y = pd.DataFrame(dataset.y)

    stratifier = IterativeStratification(n_splits=k, order=2)
    split_gen = stratifier.split(X, y)

    cv_root = os.path.join(save_root, "cvdata")
    os.makedirs(cv_root, exist_ok=True)

    for fold in tqdm(range(k), desc="Creating folds", leave=False):
        train_idx, cv_idx = next(split_gen)

        fold_dir = os.path.join(cv_root, f"fold_{fold+1}")
        train_dir = os.path.join(fold_dir, "train_data")
        cv_dir = os.path.join(fold_dir, "cv_data")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(cv_dir, exist_ok=True)

        dataset.select(train_idx.tolist(), train_dir)
        dataset.select(cv_idx.tolist(), cv_dir)

        print(f"    Fold {fold+1}: Train={len(train_idx)} CV={len(cv_idx)} saved.")


def build_full_once(merged_df: pd.DataFrame,
                    global_counter: Counter,
                    min_threshold: int,
                    use_fg: bool = True) -> tuple[dc.data.DiskDataset, list[str], str]:
    suffix = "fg_on" if use_fg else "fg_off"
    full_dir = os.path.join(FULL_ROOT, f"minthres={min_threshold}_{suffix}")
    os.makedirs(full_dir, exist_ok=True)

    dataset_dir = os.path.join(full_dir, "full_dataset")
    tasks_path = os.path.join(full_dir, "full_tasks.json")
    vector_csv = os.path.join(full_dir, "vector_full.csv")

    full_counter = Counter({k_: v for k_, v in global_counter.items() if v >= min_threshold})
    full_tasks = sorted(full_counter.keys())
    
    if len(full_tasks) == 0:
        raise RuntimeError(f"[FULL] No tasks found with min_threshold={min_threshold}")

    if os.path.exists(dataset_dir) and os.path.exists(tasks_path):
        with open(tasks_path, "r") as f:
            full_tasks = json.load(f)
        print(f"[FULL-{suffix}] Reusing existing full dataset: {dataset_dir}")
        return dc.data.DiskDataset(dataset_dir), full_tasks, full_dir

    print(f"[FULL-{suffix}] Building full tasks with min_threshold={min_threshold}, tasks={len(full_tasks)}")

    def build_label_row(row):
        d = dict.fromkeys(full_tasks, 0)
        for col in ["Descriptors", "Labels", "Odor Percepts"]:
            v = row.get(col)
            if isinstance(v, str):
                items = ast.literal_eval(v) if col == "Labels" else split_clean(v)
                for raw in items:
                    for t in extract_tokens(raw):
                        if t in d:
                            d[t] = 1
        return d

    print(f"[FULL-{suffix}] Building label matrix...")
    label_df = merged_df.progress_apply(build_label_row, axis=1, result_type="expand")
    vector_df = pd.concat([merged_df[["IsomericSMILES"]], label_df], axis=1)
    vector_df.to_csv(vector_csv, index=False)

    with open(tasks_path, "w") as f:
        json.dump(full_tasks, f, ensure_ascii=False, indent=2)

    print(f"[FULL-{suffix}] Featurizing molecules (this may take a while)...")
    featurizer = GraphFeaturizer(use_fg_features=use_fg)
    
    loader = dc.data.CSVLoader(tasks=full_tasks,
                               feature_field="IsomericSMILES",
                               featurizer=featurizer)
    
    full_dataset = loader.create_dataset([vector_csv])

    if len(full_dataset) == 0:
        raise RuntimeError(f"[FULL-{suffix}] Empty dataset after featurization")

    os.makedirs(dataset_dir, exist_ok=True)
    full_dataset.move(dataset_dir)

    print(f"[FULL-{suffix}] Saved full dataset to: {dataset_dir}")
    return dc.data.DiskDataset(dataset_dir), full_tasks, full_dir
    
if __name__ == "__main__":
    gs_df, lf_df, fldb_df = load_or_build_raw()
    GLOBAL_COUNTER = build_global_counter(gs_df, lf_df, fldb_df)
    MERGED_DF = make_merged_df(gs_df, lf_df, fldb_df)

    min_th = min(THRESHOLDS)
    target_th = 30
    use_fg_mode = False

    full_ds, full_tasks, _ = build_full_once(
        MERGED_DF,
        GLOBAL_COUNTER,
        min_th,
        use_fg=use_fg_mode
    )

    ds, tasks, ds_dir = slice_dataset_for_threshold(
        full_ds,
        full_tasks,
        GLOBAL_COUNTER,
        target_th,
        K,
        use_fg=use_fg_mode
    )

    iterative_stratified_kfold_save(ds, K, ds_dir)
