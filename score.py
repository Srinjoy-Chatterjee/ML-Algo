import numpy as np
from enum import Enum


# ==============================
# Regression
# ==============================

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def ra_score(y_true, y_pred):
    sa_res = np.sum(np.abs(y_pred - y_true))
    sa_tot = np.sum(np.abs(y_true - np.mean(y_true)))
    return 1 - (sa_res / sa_tot)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100


def mean_squared_log_error(y_true, y_pred):
    return np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)


def mean_absolute_log_error(y_true, y_pred):
    return np.mean(np.abs(np.log1p(y_pred) - np.log1p(y_true)))


# ==============================
# Classification Helpers
# ==============================

def confusion_matrix(y_true, y_pred):
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)

    for true, pred in zip(y_true, y_pred):
        true_index = np.where(unique_classes == true)[0][0]
        pred_index = np.where(unique_classes == pred)[0][0]
        matrix[true_index][pred_index] += 1

    return matrix


def _classification_components(y_true, y_pred, classification):
    cm = confusion_matrix(y_true, y_pred)

    if classification == -1:
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp
        tn = np.sum(cm) - (tp + fp + fn)
    else:
        tp = cm[classification][classification]
        fp = np.sum(cm[:, classification]) - tp
        fn = np.sum(cm[classification, :]) - tp
        tn = np.sum(cm) - (tp + fp + fn)

    return tp, fp, fn, tn


# ==============================
# Classification Metrics
# ==============================

def precision_score(y_true, y_pred, classification=-1):
    tp, fp, _, _ = _classification_components(y_true, y_pred, classification)
    precision = tp / (tp + fp + 1e-10)
    return np.mean(precision)


def recall_score(y_true, y_pred, classification=-1): #(SENSITIVITY)
    tp, _, fn, _ = _classification_components(y_true, y_pred, classification)
    recall = tp / (tp + fn + 1e-10)
    return np.mean(recall)


def specificity_score(y_true, y_pred, classification=-1):
    _, fp, _, tn = _classification_components(y_true, y_pred, classification)
    specificity = tn / (tn + fp + 1e-10)
    return np.mean(specificity)


def f1_score(y_true, y_pred, classification=-1):
    precision = precision_score(y_true, y_pred, classification)
    recall = recall_score(y_true, y_pred, classification)
    return 2 * (precision * recall) / (precision + recall + 1e-10)


def balanced_accuracy_score(y_true, y_pred, classification=-1):
    recall = recall_score(y_true, y_pred, classification)
    specificity = specificity_score(y_true, y_pred, classification)
    return (recall + specificity) / 2

def jaccard_index(y_true,y_pred):
    tp, fp, fn, _ = _classification_components(y_true, y_pred, -1)
    interaction = tp # np.sum((y_true == 1) & (y_pred == 1))
    union = tp+fp+fn # np.sum((y_true == 1) | (y_pred == 1))
    return interaction / (union + 1e-10)


def matthews_corrcoef(y_true, y_pred, classification=-1):
    tp, fp, fn, tn = _classification_components(y_true, y_pred, classification)

    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + 1e-10)


def missclassification_error(y_true, y_pred):
    return np.mean(y_true != y_pred)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def classification_report(y_true, y_pred):
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred)
    }

def log_loss(y_true,y_pred): # minimize (CROSS ENTROPY)
    """
    y_true : one-hot encoded true labels (n_samples, n_classes)
    y_pred : predicted probabilities (n_samples, n_classes)
    """
    eps = 1e-10   
    y_pred = np.clip(y_pred,eps,1-eps)
    loss = -np.mean(np.sum(y_true*np.log(y_pred),axis=1))
    return loss




# ==============================
# Enum Wrapper
# ==============================

class Score(Enum):
    R2 = r2_score
    RA = ra_score
    MSE = mean_squared_error
    MAE = mean_absolute_error
    RMSE = root_mean_squared_error
    MAPE = mean_absolute_percentage_error
    MSLE = mean_squared_log_error
    MAL = mean_absolute_log_error

    ACCURACY = accuracy_score
    PRECISION = precision_score
    RECALL = recall_score
    CLASSIFICATION_REPORT = classification_report
    F1 = f1_score
    SPECIFICITY = specificity_score
    BALANCED_ACCURACY = balanced_accuracy_score
    MCC = matthews_corrcoef
    MISSCLASSIFICATION_ERROR = missclassification_error
    LOG_LOSS = log_loss
    JACCARD_INDEX = jaccard_index

    def __call__(self, y_true, y_pred):
        return self.value(y_true, y_pred)