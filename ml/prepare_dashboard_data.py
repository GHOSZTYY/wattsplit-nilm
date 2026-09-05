"""
Consolidate the per-appliance JSON contract files (written by train_model.py)
into one compact dataset for the frontend dashboard, plus derived daily
aggregates for the anomaly/savings features.

Important data-shape note: each JSON file's power_series is the FULL aggregate
window power whenever THAT appliance was predicted ON, 0 otherwise -- it is not
a proportional per-appliance split. If two appliances are predicted ON in the
same window, naively summing their power_series would double-count the same
watts. So for any real per-appliance power estimate (line charts, pie chart,
cost calculator), this script instead uses each appliance's own REAL typical
wattage-while-on, computed from ground truth across the FULL dataset (not just
the test period) -- estimated_power = typical_wattage * predicted_on.
"""
import json
from pathlib import Path

import pandas as pd

from ground_truth import load_all_ground_truth
from ingest import load_clean_mains
from labels import METER_LABELS

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
JSON_DIR = OUT_DIR / "json_contract"
WINDOW_SECONDS = 60


def compute_typical_wattage() -> dict[str, float]:
    """Real typical ON wattage per appliance NAME, from ground truth across the
    FULL dataset span (not just the test period) -- combines meter4+meter5 into
    one 'air conditioner' figure, matching how the model itself treats them."""
    mains = load_clean_mains()
    truth = load_all_ground_truth(mains.index)

    names = sorted(set(METER_LABELS.values()) - {"mains"})
    typical = {}
    for name in names:
        meters = [m for m, n in METER_LABELS.items() if n == name]
        combined_power = truth[meters[0]]["active"].copy()
        combined_on = truth[meters[0]]["on"].copy()
        for m in meters[1:]:
            combined_power = combined_power + truth[m]["active"]
            combined_on = combined_on | truth[m]["on"]
        on_power = combined_power[combined_on]
        typical[name] = round(float(on_power.mean()), 1) if len(on_power) else 0.0
    return typical


def main():
    typical_wattage = compute_typical_wattage()
    print("typical wattage per appliance (full dataset):", typical_wattage)

    metrics = pd.read_csv(OUT_DIR / "metrics_final.csv")

    appliances = []
    timestamps = None
    states = {}

    for jf in sorted(JSON_DIR.glob("*.json")):
        payload = json.loads(jf.read_text())
        name = payload["appliance"]
        if timestamps is None:
            timestamps = payload["timestamps"]
        state_bin = [1 if s == "ON" else 0 for s in payload["predicted_state"]]
        states[name] = state_bin
        appliances.append(name)

    # daily aggregates: kWh and ON-hours per appliance per calendar day
    ts_index = pd.to_datetime(timestamps)
    days = ts_index.strftime("%Y-%m-%d")
    unique_days = sorted(set(days))

    daily = {name: {} for name in appliances}
    for name in appliances:
        state_series = pd.Series(states[name], index=ts_index)
        for day in unique_days:
            day_state = state_series[days == day]
            on_hours = day_state.sum() * WINDOW_SECONDS / 3600
            kwh = on_hours * typical_wattage[name] / 1000
            daily[name][day] = {"on_hours": round(float(on_hours), 2), "kwh": round(float(kwh), 3)}

    consolidated = {
        "window_seconds": WINDOW_SECONDS,
        "timestamps": timestamps,
        "appliances": appliances,
        "typical_wattage": typical_wattage,
        "states": states,          # 1/0 per appliance, aligned to timestamps
        "days": unique_days,
        "daily": daily,            # per-appliance per-day on_hours + kwh
        "metrics": metrics.to_dict(orient="records"),
    }

    out_path = OUT_DIR / "dashboard_data.json"
    out_path.write_text(json.dumps(consolidated))
    print(f"saved -> {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
    print(f"appliances: {appliances}")
    print(f"days covered: {unique_days}")


if __name__ == "__main__":
    main()
