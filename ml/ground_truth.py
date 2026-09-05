"""
Build ground-truth ON/OFF timelines from the real appliance submeters (meter3-12).

Unlike the aggregate signal, each submeter directly measures ONE appliance, so we
don't need edge detection here -- we just threshold each submeter's own power
reading every second: above threshold = ON, below = OFF. This becomes the labeled
"answer key" used to train and score every classifier in this project.
"""
from pathlib import Path

import pandas as pd

from ingest import H5_PATH, _load_meter
from labels import APPLIANCE_METERS

# a reading below this many Watts is treated as OFF/standby noise for that meter,
# regardless of the appliance's own typical running wattage -- small compared to
# even the smallest appliance here (water filter ~30W)
ON_THRESHOLD_W = 15.0


def load_ground_truth(
    meter_num: int, reference_index: pd.DatetimeIndex, h5_path: Path = H5_PATH
) -> pd.DataFrame:
    """Load one appliance submeter's power, reindexed onto `reference_index` (the
    SAME clean 1-second grid used for the mains signal, so predicted vs. ground
    truth line up second-for-second later).

    Important quirk of this dataset: these submeters only log a row while the
    appliance is actually drawing measurable power -- there is no explicit "0W"
    row while it's idle. So any timestamp missing from the raw data is a real OFF
    period, not a sensor gap -- it gets filled with 0, not interpolated/dropped.

    Two separate sources of NaN both need explicit filling: `reindex`'s
    fill_value only covers timestamps absent from the source index entirely
    (before this meter was installed); resample("1s") separately produces NaN
    for every idle second WITHIN the meter's own installed range (no logged row
    that second). Both mean "drawing ~0W", so both get fillna(0.0) below.
    """
    key = f"building1/elec/meter{meter_num}"
    raw = _load_meter(h5_path, key)
    power = raw["active"].resample("1s").mean()
    power = power.reindex(reference_index, fill_value=0.0).fillna(0.0)
    df = pd.DataFrame({"active": power})
    df["on"] = df["active"] > ON_THRESHOLD_W
    return df


def load_all_ground_truth(reference_index: pd.DatetimeIndex, h5_path: Path = H5_PATH) -> dict[int, pd.DataFrame]:
    """Load ground truth for every non-mains meter (3-12), aligned to `reference_index`."""
    return {m: load_ground_truth(m, reference_index, h5_path) for m in APPLIANCE_METERS}
