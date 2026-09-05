"""
Edge detection on the aggregate power signal.

Take the derivative of aggregate power, threshold it to find step changes, then debounce
(merge edges that are really one noisy transition) to get a clean list of appliance
on/off events. This is the WE Hack Round 1 checkpoint deliverable: proof the algorithm
finds real events, not disaggregation yet.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ingest import load_clean_mains


@dataclass
class EdgeParams:
    threshold_w: float = 30.0      # min |delta power| (W) between consecutive samples to count as a candidate edge
    debounce_s: int = 5            # merge candidate edges within this many seconds into one event
    settle_s: int = 3              # average power over this many seconds before/after the edge, to reject noise spikes


def detect_edges(mains: pd.DataFrame, params: EdgeParams = EdgeParams()) -> pd.DataFrame:
    """Return a DataFrame of edge events: timestamp, delta_w, direction ('rising'/'falling').

    Algorithm:
    1. Derivative: diff of 'active' power between consecutive (1s) samples.
    2. Threshold: keep samples where |diff| >= threshold_w as candidate edges.
    3. Debounce: candidate edges within `debounce_s` of each other are merged into a
       single event (keep the one with the largest |diff|) — a single physical switch
       transition often spans a couple of noisy samples.
    4. Settle check: confirm the level actually shifted by re-measuring the mean power
       in a small window before vs after the edge (rejects single-sample noise spikes
       that revert immediately).
    """
    power = mains["active"]
    diff = power.diff().fillna(0.0)

    candidates = diff[diff.abs() >= params.threshold_w]
    if candidates.empty:
        return pd.DataFrame(columns=["timestamp", "delta_w", "direction"]).set_index("timestamp")

    # debounce: greedily merge candidates within debounce_s, keep local max magnitude
    times = candidates.index.to_list()
    mags = candidates.to_list()
    merged_idx = []
    merged_val = []
    i = 0
    while i < len(times):
        j = i
        best_i = i
        while j + 1 < len(times) and (times[j + 1] - times[i]).total_seconds() <= params.debounce_s:
            j += 1
            if abs(mags[j]) > abs(mags[best_i]):
                best_i = j
        merged_idx.append(times[best_i])
        merged_val.append(mags[best_i])
        i = j + 1

    events = pd.DataFrame({"delta_w": merged_val}, index=pd.DatetimeIndex(merged_idx, name="timestamp"))

    # settle check: confirm sustained level shift, not a spike that reverts
    settle = pd.Timedelta(seconds=params.settle_s)
    keep = []
    for ts in events.index:
        before = power.loc[ts - settle: ts - pd.Timedelta(seconds=1)]
        after = power.loc[ts + pd.Timedelta(seconds=1): ts + settle]
        if before.empty or after.empty:
            keep.append(True)  # edge of data, can't verify, keep it
            continue
        sustained = abs(after.mean() - before.mean()) >= params.threshold_w * 0.5
        keep.append(sustained)
    events = events[keep]

    events["direction"] = np.where(events["delta_w"] > 0, "rising", "falling")
    return events


if __name__ == "__main__":
    mains = load_clean_mains()

    # pick a day with real activity for a readable demo plot (first calendar day is
    # meter-install day, mostly idle)
    day_str = "2013-06-07"
    day = mains.loc[day_str]
    day_events = detect_edges(day, EdgeParams())

    print(f"day={day_str}  samples={len(day)}  edges found={len(day_events)}")

    out_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 5))
    plt.plot(day.index, day["active"], linewidth=0.7, label="aggregate power")
    rising = day_events[day_events["direction"] == "rising"]
    falling = day_events[day_events["direction"] == "falling"]
    plt.scatter(rising.index, day.loc[rising.index, "active"], color="green", marker="^", s=60, label="ON event", zorder=5)
    plt.scatter(falling.index, day.loc[falling.index, "active"], color="red", marker="v", s=60, label="OFF event", zorder=5)
    plt.title(f"Aggregate power + detected edges - {day_str}")
    plt.ylabel("Watts")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / "edges_day.png"
    plt.savefig(out_path, dpi=120)
    print(f"saved plot -> {out_path}")

    events_path = out_dir / "edges_day.csv"
    day_events.to_csv(events_path)
    print(f"saved events -> {events_path}")
