"""
Window-based features and labels: the input representation for the final model.

Instead of hand-tuned edge detection deciding which moments to even look at, we
slide a fixed-length window across the ENTIRE aggregate signal and describe each
slice with simple stats. No thresholds, no clustering -- the classifier learns
each appliance's signature directly from these numbers.
"""
import pandas as pd

from labels import METER_LABELS

DEFAULT_WINDOW = "60s"


def build_window_features(power: pd.Series, window: str = DEFAULT_WINDOW) -> pd.DataFrame:
    """Per-window stats describing the shape of the aggregate signal."""
    agg = power.resample(window).agg(["mean", "std", "min", "max"])
    agg.columns = ["mean_w", "std_w", "min_w", "max_w"]
    agg["range_w"] = agg["max_w"] - agg["min_w"]
    agg["hour_of_day"] = agg.index.hour
    agg["is_weekend"] = (agg.index.dayofweek >= 5).astype(int)
    return agg.dropna()


def build_window_labels(truth: dict[int, pd.DataFrame], window: str = DEFAULT_WINDOW) -> pd.DataFrame:
    """For each appliance NAME (combining meter4+meter5 into one 'air
    conditioner' signal), the TRUE label per window is majority-vote: was it ON
    for more than half the window?"""
    names = sorted(set(METER_LABELS.values()) - {"mains"})
    labels = {}
    for name in names:
        meters = [m for m, n in METER_LABELS.items() if n == name]
        combined_on = truth[meters[0]]["on"].copy()
        for m in meters[1:]:
            combined_on = combined_on | truth[m]["on"]
        window_frac_on = combined_on.astype(float).resample(window).mean()
        labels[name] = window_frac_on > 0.5
    return pd.DataFrame(labels)


FEATURE_COLS = ["mean_w", "std_w", "min_w", "max_w", "range_w", "hour_of_day", "is_weekend"]
