"""
Load and clean the raw aggregate power signal from the iAWE dataset.

iAWE HDF5 layout: building1/elec/meter1 AND meter2 are BOTH mains meters (two-phase
supply — confirmed both by stats (both span the full 2013-05-24 to 2013-08-05
dataset range with house-scale mean/max power, unlike meter3-12 which start later
and are individual appliance sub-meters) and by the dataset's own electricity/
labels.dat file, which lists both meter1 and meter2 as "mains"). True aggregate =
meter1 + meter2.
"""
from pathlib import Path

import pandas as pd

H5_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "iawe" / "iawe.h5"

MAINS_KEYS = ["building1/elec/meter1", "building1/elec/meter2"]


def _load_meter(h5_path: Path, key: str) -> pd.DataFrame:
    df = pd.read_hdf(h5_path, key=key)
    df.columns = df.columns.droplevel(0)  # drop 'power' level, keep active/reactive/...
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def _resample_phase(df: pd.DataFrame, resample_rule: str, max_gap_s: int) -> pd.Series:
    """Resample one phase to a uniform grid, interpolating gaps up to max_gap_s."""
    s = df["active"].resample(resample_rule).mean()
    s = s.interpolate(method="time", limit=max_gap_s)
    return s


def load_clean_mains(h5_path: Path = H5_PATH, resample_rule: str = "1s", max_gap_s: int = 30) -> pd.DataFrame:
    """Load both mains phases, each resampled to a common uniform grid, then sum.

    Resampling each phase onto the same grid *before* summing avoids falsely zeroing
    out real power just because the two phase meters' raw sample timestamps don't
    line up exactly (they're independent sensors, not synchronized to the millisecond).
    Rows where both phases are still NaN after gap interpolation (a real dropout) are
    dropped.
    """
    phases = [_load_meter(h5_path, key) for key in MAINS_KEYS]
    resampled = [_resample_phase(p, resample_rule, max_gap_s) for p in phases]
    active = resampled[0].add(resampled[1], fill_value=0)
    both_missing = resampled[0].isna() & resampled[1].isna()
    active = active[~both_missing.reindex(active.index, fill_value=False)]
    return pd.DataFrame({"active": active}).dropna(subset=["active"])


if __name__ == "__main__":
    mains = load_clean_mains()
    print(f"rows={len(mains)}  span={mains.index.min()} -> {mains.index.max()}")
    print(f"active power: mean={mains['active'].mean():.1f}W  max={mains['active'].max():.1f}W")
