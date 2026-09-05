"""
Metric helpers shared by train_model.py: per-appliance decision threshold
tuning and precision/recall/F1 scoring.

Threshold tuning matters here because a plain 0.5 cutoff misses almost every
real event for a rare appliance (its predicted probability rarely climbs that
high). Thresholds are tuned on a VALIDATION slice carved from the training
period, never the real test set, so reported test scores stay genuinely held-out.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

THRESHOLDS = np.arange(0.05, 0.95, 0.05)


def best_threshold(y_val: pd.Series, proba_val: np.ndarray) -> float:
    """Sweep thresholds on VALIDATION data only, return whichever maximizes F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLDS:
        f1 = f1_score(y_val, proba_val >= t, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def score(y_test: pd.Series, pred_test: np.ndarray) -> dict:
    """Precision/recall/F1 for one appliance's test predictions."""
    return {
        "precision": round(precision_score(y_test, pred_test, zero_division=0), 2),
        "recall": round(recall_score(y_test, pred_test, zero_division=0), 2),
        "f1": round(f1_score(y_test, pred_test, zero_division=0), 2),
    }


def fair_macro_f1(results: list[dict], all_appliance_count: int) -> float:
    """Macro F1 averaged over ALL appliances that exist (including ones with no
    usable model at all, scored as 0) -- not just the ones a model was built
    for. Averaging only over modeled appliances quietly inflates the number by
    dropping the hardest cases from the denominator."""
    total_f1 = sum(r.get("f1", 0.0) for r in results)
    return total_f1 / all_appliance_count
