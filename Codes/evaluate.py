import os
import json
import random
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import deepchem as dc

def cosine_similarity_centered(y_true, y_pred):
    y_true = np.atleast_2d(y_true)
    y_pred = np.atleast_2d(y_pred)
    y_true_centered = y_true
    y_pred_centered = y_pred - np.mean(y_pred, axis=1, keepdims=True)
    dot = np.sum(y_true_centered * y_pred_centered, axis=1)
    norm_true = np.linalg.norm(y_true_centered, axis=1)
    norm_pred = np.linalg.norm(y_pred_centered, axis=1)

    cosine_sim = dot / (norm_true * norm_pred + 1e-8)
    return np.mean(cosine_sim)

def mean_average_precision(y_true, y_pred):
    return average_precision_score(y_true, y_pred, average='macro')

def micro_roc_auc_score(y_true, y_pred):
    return dc.metrics.roc_auc_score(y_true, y_pred, average='micro')

def macro_roc_auc_score(y_true, y_pred):
    return dc.metrics.roc_auc_score(y_true, y_pred, average='macro')

def pr_auc_score_micro(y_true, y_pred):
    return average_precision_score(y_true, y_pred, average="micro")

def pr_auc_score_macro(y_true, y_pred):
    return average_precision_score(y_true, y_pred, average="macro")

def evaluate_model(model, dataset):
    y_pred = model.predict(dataset)
    y_true = dataset.y
    
    results = {
        "ROC-AUC(micro)": micro_roc_auc_score(y_true, y_pred),
        "ROC-AUC(macro)": macro_roc_auc_score(y_true, y_pred),
        "PR-AUC(micro)": pr_auc_score_micro(y_true, y_pred),   
        "PR-AUC(macro)": pr_auc_score_macro(y_true, y_pred), 
        "CosineSimilarity": cosine_similarity_centered(y_true, y_pred),
    }
    return results

