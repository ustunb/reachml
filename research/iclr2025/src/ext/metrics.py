"""
this file contains functions to compute standard classification metrics
and run hypothesis tests to compare performance differences
"""
import numpy as np
from scipy import stats as st
from sklearn.utils import resample
from sklearn.metrics import log_loss

def compute_error(y_true, y_pred):
    """
    computes error rate for a binary classifier
    :param y_true:
    :param y_pred: predictions
    :return:
    """
    return np.not_equal(y_true, y_pred).mean()

def compute_log_loss(y_true, y_pred, eps=1e-15):
    """
    computes mean logistic loss using labels and probability predictions
    :param y_true: vector of true labels - y_true[i] in (-1,+1) and y_true[i] (0,1) are both OK
    :param y_pred: vector of predicted probabilities
    :param eps: minimum distance that y_pred must maintain from 0 and 1
    :return: normalized logistic loss where:
              y = y_true > 0
              L = y * np.log(p) + (1 - y) * np.log(1.0 - p)
              log_loss = np.mean(L)
    -----------------------
    Note that the implemntation is meant to be fast, but isn't easy to understand.
    Here are two other implementations to clarify what's going on
    ##### basic #####
    p = np.clip(y_pred, eps, 1.0 - eps)
    y = y_true > 0
    L = y * np.log(p) + (1 - y) * np.log(1.0 - p)
    return np.mean(L)
    ##### faster (only calls log once) #####
    p = np.clip(y_pred, eps, 1.0 - eps) # clip probabilities
    I_pos = y_true > 0                  # convert y_true to y in (0,1) /indices of true points
    L = np.empty_like(y_true)
    L[idx_pos] = p[I_pos]
    L[~idx_pos] = 1 - p[~I_pos]
    return np.log(L).mean()
    """
    #### fastest implementation (only calls log once / minimizes storage)

    # L will eventually contain the values of the log-loss at each point

    # Initialize L = p
    L = np.clip(y_pred, eps, 1.0 - eps)

    # Let I_neg = i where y_true[i] == 0
    I_neg = y_true < 1

    # Set L[i] = 1 - p[i] for all i where y_true[i] ≠ 1
    # L[I_neg] -> -L[I_neg] -> 1 - L[I_neg]
    L[I_neg] *= -1
    L[I_neg] += 1

    return -1 * np.log(L).mean()

def compute_auc(y_true, y_pred):
    """
    computes AUC for a binary classifier quickly
    :param y_true: vector of true classes
    :param y_pred: vector of predicted probabilities
    :return: auc
    """
    n = len(y_true)
    I_pos = y_true > 0
    # if y[i] == 0 for all i or y[i] == 1 for all i, then return AUC = 1
    if I_pos.all() or np.logical_not(I_pos).all():
        return float('nan')

    I_pos = I_pos[np.argsort(y_pred)]
    false_positive_count = np.cumsum(1 - I_pos)
    n_false = false_positive_count[-1]
    auc = np.multiply(I_pos, false_positive_count).sum() / (
            n_false * (n - n_false))
    return auc

def compute_ece(y_true, y_pred, n_bins=10):
    """
    computes the expected calibration error quickly
    :param y_true: vector of true labels
    :param y_pred: vector of predicted probabilities
    :param n_bins: 10
    :return: expected calibration error (L1)
    """
    # pre-process to improve binning
    sort_idx = np.argsort(y_pred)
    ys_prob = y_pred[sort_idx]
    ys_diff = ys_prob - (y_true[sort_idx] > 0)
    err = 0.0
    i = 0
    for k in range(1, n_bins + 1):
        j = np.searchsorted(ys_prob, k / n_bins, side='right')
        if i < j:
            err += np.abs(np.sum(ys_diff[i:j]))
        i = j
    ece = err / len(y_true)

    # try:
    #     assert np.isclose(ece, ece_score_basic(y_true, y_pred))
    #     assert np.isclose(ece, ece_score_fast(y_true, y_pred))
    # except AssertionError:
    #     from ext.debug import ipsh
    #     ipsh()

    return ece
