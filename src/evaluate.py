"""Metrics, calibration, and cost-based threshold selection."""

import numpy as np
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def pr_auc(y_true, y_score):
    return average_precision_score(y_true, y_score)


def roc_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)


def gini(y_true, y_score):
    return 2 * roc_auc_score(y_true, y_score) - 1


def ks_statistic(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    return ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic


def evaluate_all(y_true, y_score):
    return {
        "PR_AUC": pr_auc(y_true, y_score),
        "ROC_AUC": roc_auc(y_true, y_score),
        "Gini": gini(y_true, y_score),
        "KS": ks_statistic(y_true, y_score),
    }


def brier(y_true, y_prob):
    return brier_score_loss(y_true, y_prob)


def reliability_points(y_true, y_prob, n_bins=10):
    """Return (mean_predicted, observed_fraction) per bin for a reliability diagram."""
    observed, predicted = calibration_curve(y_true, y_prob, n_bins=n_bins,
                                             strategy="quantile")
    return predicted, observed


def applicant_costs(amt_credit, amt_annuity, lgd=0.6):
    c_fn = lgd * np.asarray(amt_credit, dtype=float)
    c_fp = np.asarray(amt_annuity, dtype=float)
    return c_fn, c_fp


def confusion_at(y_true, pd_proba, threshold):
    """Count TN, FP, FN, TP when declining everyone with PD >= threshold."""
    y_true = np.asarray(y_true)
    decline = np.asarray(pd_proba) >= threshold
    tp = int(((y_true == 1) & decline).sum())
    fn = int(((y_true == 1) & ~decline).sum())
    fp = int(((y_true == 0) & decline).sum())
    tn = int(((y_true == 0) & ~decline).sum())
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def sweep_threshold(y_true, pd_proba, c_fn, c_fp, thresholds=None):
    """At each threshold, total the cost of the mistakes and track approval rate."""
    y_true = np.asarray(y_true)
    pd_proba = np.asarray(pd_proba)
    c_fn = np.asarray(c_fn, dtype=float)
    c_fp = np.asarray(c_fp, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    costs, approval = [], []
    for t in thresholds:
        decline = pd_proba >= t
        fn_cost = c_fn[(y_true == 1) & ~decline].sum()
        fp_cost = c_fp[(y_true == 0) & decline].sum()
        costs.append(fn_cost + fp_cost)
        approval.append(float((~decline).mean()))
    return {
        "thresholds": np.asarray(thresholds),
        "cost": np.asarray(costs),
        "approval_rate": np.asarray(approval),
    }


def pick_threshold(y_true, pd_proba, c_fn, c_fp, thresholds=None):
    """Return the threshold t* that minimizes total expected cost."""
    sweep = sweep_threshold(y_true, pd_proba, c_fn, c_fp, thresholds)
    best = int(np.argmin(sweep["cost"]))
    return float(sweep["thresholds"][best])
