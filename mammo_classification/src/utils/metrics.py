from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

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
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'kappa': cohen_kappa_score(y_true, y_pred)
    }
    return metrics