"""
K-means clustering of edge magnitudes -> crude per-appliance disaggregation.

Idea: every real appliance has a roughly consistent power draw (fridge ~150W,
AC ~1500-2000W, iron ~1000W, ...). So if we cluster the *sizes* of the edges we
detected (ignoring direction), edges from the same appliance should land in the
same cluster regardless of which specific ON/OFF event they were. Once edges are
clustered, we pair up a rising edge with the next falling edge in the SAME cluster
to reconstruct "appliance X was ON from time A to time B" intervals.

This is intentionally crude (no supervised ML) -- it is the WE Hack Round 2
checkpoint: "crude disaggregation working, visible per-appliance split." The
project's final, accurate model (train_model.py) supersedes this for real
numbers -- this stays as the Round 2 proof-of-concept artifact.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from edge_detection import EdgeParams, detect_edges
from ingest import load_clean_mains
from labels import EXPECTED_WATTAGE


@dataclass
class Interval:
    cluster: int
    appliance_guess: str
    start: pd.Timestamp
    end: pd.Timestamp
    magnitude_w: float


def choose_k(magnitudes_log: np.ndarray, k_range=range(6, 11)) -> int:
    """Pick the number of clusters via silhouette score, searched only in the
    6-10 range: the dataset's own labels.dat lists 9 distinct non-mains
    appliance names, so letting silhouette search from k=2 tends to pick an
    artificially low k that lumps several unrelated appliances of similar
    wattage into one cluster -- which then corrupts the rising/falling pairing
    step downstream. Falls back to k=8 if scoring is degenerate."""
    best_k, best_score = 8, -1.0
    X = magnitudes_log.reshape(-1, 1)
    for k in k_range:
        if k >= len(magnitudes_log):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        score = silhouette_score(X, km.labels_)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def cluster_edges(events: pd.DataFrame, k: int | None = None) -> tuple[pd.DataFrame, dict[int, float]]:
    """Cluster edges by |delta_w| (log-scaled so a 150W fridge jump and an 8000W
    AC jump don't get squashed onto the same scale)."""
    magnitude = events["delta_w"].abs()
    log_mag = np.log1p(magnitude.values)

    if k is None:
        k = choose_k(log_mag)

    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(log_mag.reshape(-1, 1))
    events = events.copy()
    events["cluster"] = km.labels_

    centroid_w = {c: float(np.expm1(km.cluster_centers_[c][0])) for c in range(k)}
    return events, centroid_w


def guess_appliance_names(centroid_w: dict[int, float]) -> dict[int, str]:
    """For each cluster centroid (a wattage), pick the closest known appliance's
    expected wattage as a human-readable guess -- a friendly label for the demo
    plot, not a real classification."""
    guesses = {}
    for cluster, watts in centroid_w.items():
        closest = min(EXPECTED_WATTAGE, key=lambda name: abs(EXPECTED_WATTAGE[name] - watts))
        guesses[cluster] = closest
    return guesses


def pair_edges_into_intervals(
    events: pd.DataFrame, appliance_guess: dict[int, str], max_gap_s: int = 3600
) -> list[Interval]:
    """For each cluster, walk events in time order. A 'rising' edge opens an
    interval; the next 'falling' edge in the same cluster closes the most
    recently opened one (LIFO). max_gap_s caps how long a rising edge can sit
    open before being discarded as unmatched (default 1 hour) -- without this
    cap, an unmatched rising edge would eventually pair with some unrelated
    falling edge hours later, producing a fake "appliance ran all day" interval."""
    intervals: list[Interval] = []
    open_stacks: dict[int, list[pd.Timestamp]] = {}

    for ts, row in events.sort_index().iterrows():
        cluster = int(row["cluster"])
        stack = open_stacks.setdefault(cluster, [])
        if row["direction"] == "rising":
            stack.append(ts)
        else:  # falling
            if stack:
                start = stack.pop()
                if (ts - start).total_seconds() <= max_gap_s:
                    intervals.append(Interval(
                        cluster=cluster, appliance_guess=appliance_guess[cluster],
                        start=start, end=ts, magnitude_w=abs(row["delta_w"]),
                    ))
    return intervals


if __name__ == "__main__":
    mains = load_clean_mains()
    week = mains.loc["2013-06-07":"2013-06-13"]
    events = detect_edges(week, EdgeParams())
    print(f"edges found over week: {len(events)}")

    events, centroid_w = cluster_edges(events)
    appliance_guess = guess_appliance_names(centroid_w)

    print("\ncluster centroids (guessed appliance):")
    for c in sorted(centroid_w):
        print(f"  cluster {c}: ~{centroid_w[c]:.0f}W  -> guess: {appliance_guess[c]}")

    intervals = pair_edges_into_intervals(events, appliance_guess)
    print(f"\npaired ON intervals: {len(intervals)}")

    out_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    intervals_df = pd.DataFrame([{
        "appliance_guess": iv.appliance_guess, "cluster": iv.cluster,
        "start": iv.start, "end": iv.end,
        "duration_s": (iv.end - iv.start).total_seconds(), "magnitude_w": iv.magnitude_w,
    } for iv in intervals])
    intervals_df.to_csv(out_dir / "disaggregation_intervals.csv", index=False)
    print(f"saved -> {out_dir / 'disaggregation_intervals.csv'}")

    # Gantt-style plot: one day, one row per guessed appliance, bars = ON intervals
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    day_str = "2013-06-07"
    day = week.loc[day_str]
    day_start = day.index.min()
    day_end = day.index.max()

    day_intervals = intervals_df[
        (intervals_df["start"] >= day_start) & (intervals_df["start"] <= day_end)
    ]

    fig, (ax_power, ax_gantt) = plt.subplots(
        2, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
    )

    # Both panels stay on the same plain numeric "hours since midnight" x-axis.
    # (Plotting real timestamps on one panel and plain floats on the other, with
    # sharex=True, makes matplotlib misinterpret the numeric floats as dates.)
    day_hours = (day.index - day_start).total_seconds() / 3600
    ax_power.plot(day_hours, day["active"], linewidth=0.7, color="steelblue")
    ax_power.set_ylabel("Watts")
    ax_power.set_title(f"Aggregate power - {day_str}")

    appliances = sorted(day_intervals["appliance_guess"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(appliances)))
    color_map = dict(zip(appliances, colors))

    for i, appliance in enumerate(appliances):
        rows = day_intervals[day_intervals["appliance_guess"] == appliance]
        for _, r in rows.iterrows():
            ax_gantt.barh(
                i, (r["end"] - r["start"]).total_seconds() / 3600,
                left=(r["start"] - day_start).total_seconds() / 3600,
                height=0.6, color=color_map[appliance],
            )
    ax_gantt.set_yticks(range(len(appliances)))
    ax_gantt.set_yticklabels(appliances)
    ax_gantt.set_xlabel("Hours since midnight")
    ax_gantt.set_title("Crude disaggregation: guessed appliance ON intervals")
    ax_gantt.set_xlim(0, 24)

    plt.tight_layout()
    out_path = out_dir / "disaggregation_gantt.png"
    plt.savefig(out_path, dpi=120)
    print(f"saved plot -> {out_path}")
