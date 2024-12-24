def accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

def precision(y_true, y_pred):
    true_positives = sum((y_true == 1) & (y_pred == 1))
    predicted_positives = sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives > 0 else 0

def recall(y_true, y_pred):
    true_positives = sum((y_true == 1) & (y_pred == 1))
    actual_positives = sum(y_true == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0