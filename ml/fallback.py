"""
Fallback Layer 2: rule-based combinatorial matching, no ML required.

If the trained model isn't converging under hackathon time pressure, this is
the safety net. Idea: at every timestep, brute-force try every combination of
"which known appliances are ON" and see which combination's total wattage sum
lands closest to the actually observed aggregate power. With ~7 appliances
that's only 2^7 = 128 combinations to check per second -- fast enough to
vectorize with numpy.
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from ground_truth import load_all_ground_truth
from ingest import load_clean_mains
from labels import METER_LABELS

# appliances usable for this fallback: skip meter9/11 (kitchen outlets, water
# filter) since they weren't installed yet during our demo week and have no
# ground truth to calibrate a typical wattage from
FALLBACK_METERS = [3, 4, 5, 6, 7, 8, 10]


def typical_wattage_per_appliance(truth: dict[int, pd.DataFrame]) -> dict[int, float]:
    """Real typical ON wattage per appliance, computed from ITS OWN ground-truth
    data (mean power while that appliance's ON flag is True)."""
    watts = {}
    for meter in FALLBACK_METERS:
        on_power = truth[meter]["active"][truth[meter]["on"]]
        watts[meter] = on_power.mean() if len(on_power) else 0.0
    return watts


def build_combo_table(wattages: dict[int, float]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """All 2^n ON/OFF combinations of the given appliances, and each combo's
    total wattage sum."""
    meters = list(wattages.keys())
    w = np.array([wattages[m] for m in meters])
    combos = np.array(list(product([0, 1], repeat=len(meters))))  # shape (2^n, n)
    combo_sums = combos @ w  # shape (2^n,)
    return combos, combo_sums, meters


def disaggregate_rule_based(aggregate_power: pd.Series, wattages: dict[int, float]) -> pd.DataFrame:
    """For every timestep, pick whichever ON/OFF combination's total wattage is
    closest to the observed aggregate reading. Returns one boolean ON/OFF column
    per appliance meter (keyed by METER NUMBER, since meter4 and meter5 are both
    "air conditioner" and would collide if keyed by name)."""
    combos, combo_sums, meters = build_combo_table(wattages)

    observed = aggregate_power.to_numpy().reshape(-1, 1)          # (N, 1)
    diffs = np.abs(observed - combo_sums.reshape(1, -1))          # (N, 2^n)
    best_combo_idx = diffs.argmin(axis=1)                          # (N,)
    chosen = combos[best_combo_idx]                                # (N, n)

    return pd.DataFrame(chosen.astype(bool), index=aggregate_power.index, columns=meters)


if __name__ == "__main__":
    day_str = "2013-06-07"
    mains = load_clean_mains()
    day = mains.loc[day_str]

    reference_index = day.index
    truth = load_all_ground_truth(reference_index)
    wattages = typical_wattage_per_appliance(truth)

    print("typical wattage per appliance (from ground truth):")
    for m in FALLBACK_METERS:
        print(f"  {METER_LABELS[m]:<16}: {wattages[m]:.1f}W")

    predicted = disaggregate_rule_based(day["active"], wattages)

    print("\nper-appliance agreement with ground truth (accuracy, ON vs OFF each second):")
    for m in FALLBACK_METERS:
        actual_on = truth[m]["on"].reindex(predicted.index, fill_value=False)
        agreement = (predicted[m] == actual_on).mean()
        print(f"  meter{m:>2} ({METER_LABELS[m]:<16}): {agreement * 100:5.1f}%")

    out_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    predicted_named = predicted.rename(columns={m: f"meter{m}_{METER_LABELS[m].replace(' ', '_')}" for m in predicted.columns})
    predicted_named.to_csv(out_dir / "fallback_rule_based_predictions.csv")
    print(f"\nsaved -> {out_dir / 'fallback_rule_based_predictions.csv'}")
