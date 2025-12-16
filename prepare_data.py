import os
import pandas as pd
import pyrfume
import math
from collections import Counter, OrderedDict
import ast
import matplotlib.pyplot as plt
import pickle
import json
import seaborn as sns
from graph_featurizer import GraphFeaturizer
import deepchem as dc
from openpom.feat.graph_featurizer import GraphConvConstants
from openpom.utils.data_utils import get_class_imbalance_ratio, IterativeStratifiedSplitter
from openpom.models.mpnn_pom import MPNNPOMModel
from datetime import datetime
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import pickle
import json


prefix = "PYRFUME_DATA_DIR"

raw_path = "../data/raw"
dataset_path = "../data/dataset"
preprocessed_path = "../data/preprocessed"
os.makedirs(raw_path, exist_ok=True)
os.makedirs(preprocessed_path, exist_ok=True)

goodscents_path = os.path.join(raw_path, "goodscents_merged.csv")
leffingwell_path = os.path.join(raw_path, "leffingwell_merged.csv")
flavordb_path = os.path.join(raw_path, "flavordb_merged.csv")

def safe_load(path):
    full = os.path.join(prefix, path)
    if not os.path.exists(full):
        raise FileNotFoundError(full)
    return pyrfume.load_data(full, remote=False).reset_index()

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

gs_df = pd.read_csv(goodscents_path)
lf_df = pd.read_csv(leffingwell_path)
fldb_df = pd.read_csv(flavordb_path)

counter_df = Counter()
for i in gs_df["Descriptors"]:
    if isinstance(i, str):
        counter_df.update(i.split(";"))     

counter_lf = Counter()
for i in lf_df["Labels"]:
    if isinstance(i, str):
        counter_lf.update(ast.literal_eval(i))  

counter_fldb = Counter()
for i in fldb_df["Flavor Percepts"]:
    if isinstance(i, str):
        counter_fldb.update(i.split(";"))     

all_counter = Counter()
all_counter.update(counter_df)
all_counter.update(counter_lf)
all_counter.update(counter_fldb)

thres = 30
flavor_file = "flavor_list" + "_thres=" + str(thres) + ".json"
flavor_path = os.path.join(preprocessed_path, flavor_file)

filtered_counts = Counter({k: v for k, v in all_counter.items() if v >= thres})
with open(flavor_path, "w") as f:
    json.dump(list(filtered_counts.keys()), f)

gs_df = gs_df.rename(columns={"IsomericSMILES": "IsomericSMILES_gs"})
lf_df = lf_df.rename(columns={"IsomericSMILES": "IsomericSMILES_lf"})
fldb_df = fldb_df.rename(columns={"IsomericSMILES": "IsomericSMILES_fldb"})

merged_df = gs_df.merge(lf_df, on="CID", how="outer", suffixes=('_gs', '_lf'))
merged_df = merged_df.merge(fldb_df, on="CID", how="outer", suffixes=('', '_fldb'))

def resolve_smiles(row):
    s = {row['IsomericSMILES_gs'], row['IsomericSMILES_lf'], row['IsomericSMILES_fldb']}
    s = {x for x in s if not (isinstance(x, float) and math.isnan(x))}
    s.discard(None)
    return "; ".join(s) if len(s) > 1 else next(iter(s), None)

merged_df["Resolved_IsomericSMILES"] = merged_df.apply(resolve_smiles, axis=1)
merged_df.drop(columns=['IsomericSMILES_gs', 'IsomericSMILES_lf', 'IsomericSMILES_fldb'], inplace=True)
merged_df.rename(columns={'Resolved_IsomericSMILES': 'IsomericSMILES'}, inplace=True)

def process_flavor(row, flavor_lists):
    d = dict.fromkeys(flavor_lists, 0)
    if isinstance(row["Descriptors"], str):
        for k in row["Descriptors"].split(";"):
            if k in d:
                d[k] = 1
    if isinstance(row["Labels"], str):
        for k in ast.literal_eval(row["Labels"]):
            if k in d:
                d[k] = 1
    if isinstance(row["Flavor Percepts"], str):
        for k in row["Flavor Percepts"].split(";"):
            if k in d:
                d[k] = 1
    return d

with open(flavor_path, "r") as f:
    flavor_lists = json.load(f)

vector_df = merged_df.apply(process_flavor, args=(flavor_lists,), axis=1, result_type='expand')
vector_df = pd.concat([merged_df, vector_df], axis=1)

columns_to_drop = [
    'Stimulus_gs', 'TGSC ID', 'CID', 'Concentration %', 'Solvent', 'Descriptors',
    'MolecularWeight_gs', 'IUPACName_gs', 'name_gs', 'Stimulus_lf', 'MolecularWeight_lf',
    'IUPACName_lf', 'name_lf', 'cas', 'Raw Labels', 'Labels', 'Stimulus',
    'Odor Percepts', 'Odor Modifiers', 'Flavor Percepts', 'Flavor Modifiers',
    'MolecularWeight', 'IUPACName', 'name', 'threshold'
]
vector_df = vector_df.drop(columns=columns_to_drop, errors='ignore')
vector_df.to_csv(os.path.join(preprocessed_path, f"vector_data_frethres={thres}.csv"), index=False)

input_file = os.path.join(preprocessed_path, f"vector_data_frethres={thres}.csv")
n_tasks = len(flavor_lists)
featurizer = GraphFeaturizer()
loader = dc.data.CSVLoader(
    tasks=flavor_lists,
    feature_field="IsomericSMILES",
    featurizer=featurizer
)

dataset = loader.create_dataset(inputs=[input_file])
dataset.move(os.path.join(dataset_path, f"frethres={thres}"))