from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def calculate_metrics(y_true, y_pred):
    """
    Calculate various classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        dict: Dictionary containing various metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted',zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred)
    }
    return metrics

def calculate_class_weights(dataset):
    """Calculate class weights to handle imbalanced data"""
    labels = [dataset.label_to_idx[label] for label in dataset.df[dataset.task]]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.FloatTensor(class_weights).cuda()