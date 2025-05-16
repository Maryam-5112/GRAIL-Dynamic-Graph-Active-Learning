
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, average_precision_score


# Function to calculate custom metrics for binary classification
def custom_metrics_binary(true_labels, pred_probs, metrics_list=None):
    if metrics_list is None:
        metrics_list = ['accuracy', 'precision', 'recall', 'roc_auc', 'pr_auc', 'f1_micro', 'f1_macro']

    pred_labels = (pred_probs > 0.5).astype(int)
    
    results = {}
   

    # Compute the metrics based on the provided list
    if 'accuracy' in metrics_list:
        results['accuracy'] = accuracy_score(true_labels, pred_labels)

    if 'precision' in metrics_list:
        results['precision'] = precision_score(true_labels, pred_labels, zero_division=0)

    if 'recall' in metrics_list:
        results['recall'] = recall_score(true_labels, pred_labels, zero_division=0)

    if 'roc_auc' in metrics_list:
        if len(set(true_labels)) > 1:
            results['ROC_AUC'] = roc_auc_score(true_labels, pred_probs)

    if 'pr_auc' in metrics_list:
        if len(set(true_labels)) > 1:
            results['AUC_PR'] = average_precision_score(true_labels, pred_probs)

    if 'f1_micro' in metrics_list:
        results['f1_micro'] = f1_score(true_labels, pred_labels, average='micro', zero_division=0)

    if 'f1_macro' in metrics_list:
        results['f1_macro'] = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

    return results


# Function to calculate custom metrics for multi-class classification
def custom_metrics_multiclass(true_labels, pred_probs, metrics_list=None):
    if metrics_list is None:
        metrics_list = ['accuracy', 'precision', 'recall', 'f1_micro', 'f1_macro']

    pred_labels = pred_probs.argmax(axis=1)
    
    results = {}

    # Compute the metrics based on the provided list
    if 'accuracy' in metrics_list:
        results['accuracy'] = accuracy_score(true_labels, pred_labels)

    if 'precision' in metrics_list:
        results['precision'] = precision_score(true_labels, pred_labels, average='macro', zero_division=0)

    if 'recall' in metrics_list:
        results['recall'] = recall_score(true_labels, pred_labels, average='macro', zero_division=0)

    if 'f1_micro' in metrics_list:
        results['f1_micro'] = f1_score(true_labels, pred_labels, average='micro', zero_division=0)

    if 'f1_macro' in metrics_list:
        results['f1_macro'] = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

    return results
