"""LightGBM model builders for the credit-risk project."""
from lightgbm import LGBMClassifier


def pos_weight(y):
    """scale_pos_weight = (# non-defaults) / (# defaults)."""
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return n_neg / n_pos


def make_lgbm(scale_pos_weight, n_estimators=400, learning_rate=0.05):
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=50,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
