"""
Meter number -> appliance name mapping, straight from the iAWE dataset's own
electricity/labels.dat file (extracted from electricity.tar.gz), not guessed.
"""

METER_LABELS = {
    1: "mains",
    2: "mains",
    3: "fridge",
    4: "air conditioner",
    5: "air conditioner",
    6: "washing machine",
    7: "laptop computer",
    8: "iron",
    9: "kitchen outlets",
    10: "television",
    11: "water filter",
    12: "water motor",
}

MAINS_METERS = [m for m, name in METER_LABELS.items() if name == "mains"]
APPLIANCE_METERS = [m for m, name in METER_LABELS.items() if name != "mains"]

# rough expected steady-state power draw per appliance (Watts), used only as a
# first-guess label for K-means clusters in clustering.py's crude disaggregation
# demo -- NOT used by the final model (train_model.py), which learns real
# per-appliance detectors from ground truth instead of guessing from typical wattage
EXPECTED_WATTAGE = {
    "fridge": 150,
    "air conditioner": 1500,
    "washing machine": 400,
    "laptop computer": 60,
    "iron": 1000,
    "kitchen outlets": 500,
    "television": 80,
    "water filter": 30,
    "water motor": 250,
}
