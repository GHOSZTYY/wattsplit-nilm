"""
FINAL locked-in pipeline for WattSplit's disaggregation model.

  Aggregate signal -> 60s window features (mean/std/min/max/range/hour/weekend)
  -> one binary RandomForestClassifier per appliance, class_weight='balanced'
  -> per-appliance decision threshold tuned on a held-out validation slice
  -> predict on the true test period

This architecture was chosen after comparing 7 alternatives: K-means clustering
+ edge pairing (macro F1 0.235), per-edge classification (0.331), an FHMM
academic baseline (0.078), and plain window-based RandomForest/Gradient
Boosting at 60s and 20s windows (0.33-0.38 fair). Per-appliance threshold
tuning on this specific train/validation/test split won clearly: verified
macro F1 = 0.475 (raw, over 8 modeled appliances) / 0.422 (fair, over all 9
including water motor, which has zero trainable data in this split and is
excluded from the model entirely -- not scored as a false success).

Run this to both print final metrics AND regenerate the JSON output contract
for Backend (one file per appliance, covering the real test period).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from features import FEATURE_COLS, build_window_features, build_window_labels
from ground_truth import load_all_ground_truth
from ingest import load_clean_mains
from metrics import best_threshold, fair_macro_f1, score

WINDOW = "60s"

TRAIN_START, TRAIN_END = "2013-05-24", "2013-07-12"
VAL_START, VAL_END = "2013-07-13", "2013-07-27"
TEST_START, TEST_END = "2013-07-28", "2013-08-05"


if __name__ == "__main__":
    mains = load_clean_mains()
    full_window = mains.loc[TRAIN_START:TEST_END]
    X = build_window_features(full_window["active"], window=WINDOW)
    truth = load_all_ground_truth(full_window.index)
    Y = build_window_labels(truth, window=WINDOW).reindex(X.index).fillna(False)

    tz = X.index.tz
    train_mask = X.index < pd.Timestamp(VAL_START, tz=tz)
    val_mask = (X.index >= pd.Timestamp(VAL_START, tz=tz)) & (X.index < pd.Timestamp(TEST_START, tz=tz))
    test_mask = X.index >= pd.Timestamp(TEST_START, tz=tz)

    X_train, X_val, X_test = X.loc[train_mask, FEATURE_COLS], X.loc[val_mask, FEATURE_COLS], X.loc[test_mask, FEATURE_COLS]
    Y_train, Y_val, Y_test = Y.loc[train_mask], Y.loc[val_mask], Y.loc[test_mask]
    test_agg_power = full_window["active"].resample(WINDOW).mean().reindex(X_test.index)

    print(f"window={WINDOW}  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    out_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    json_dir = out_dir / "json_contract"
    json_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_appliances = sorted(Y.columns)

    for name in all_appliances:
        y_train, y_val, y_test = Y_train[name], Y_val[name], Y_test[name]
        if y_train.sum() == 0:
            print(f"  {name}: skipped, zero positive training windows -- no model, no JSON file")
            rows.append({"appliance": name, "threshold": None, "precision": 0.0, "recall": 0.0, "f1": 0.0, "modeled": False})
            continue

        clf = RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced")
        clf.fit(X_train, y_train)

        t = best_threshold(y_val, clf.predict_proba(X_val)[:, 1])
        proba_test = clf.predict_proba(X_test)[:, 1]
        pred_test = proba_test >= t

        result = score(y_test, pred_test)
        result.update({"appliance": name, "threshold": round(t, 2), "modeled": True})
        rows.append(result)
        print(f"  {name:<16} threshold={t:.2f}  precision={result['precision']:.2f}  "
              f"recall={result['recall']:.2f}  f1={result['f1']:.2f}")

        # JSON output contract: power_series = real aggregate window power while
        # predicted ON, 0 elsewhere; predicted_state mirrors that as ON/OFF
        power_out = test_agg_power.where(pred_test, 0.0)
        state_out = np.where(pred_test, "ON", "OFF")
        payload = {
            "appliance": name,
            "timestamps": X_test.index.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            "power_series": power_out.round(1).tolist(),
            "predicted_state": state_out.tolist(),
        }
        safe_name = name.replace(" ", "_")
        with open(json_dir / f"{safe_name}.json", "w") as f:
            json.dump(payload, f)

    results_df = pd.DataFrame(rows)
    modeled = results_df[results_df["modeled"]]

    print(f"\nraw macro F1 (over {len(modeled)} modeled appliances): {modeled['f1'].mean():.3f}")
    print(f"fair macro F1 (over all {len(results_df)} appliances, unmodeled = 0): "
          f"{fair_macro_f1(rows, len(results_df)):.3f}")

    results_df.to_csv(out_dir / "metrics_final.csv", index=False)
    print(f"\nsaved metrics -> {out_dir / 'metrics_final.csv'}")
    print(f"saved JSON contract -> {json_dir}/ ({len(modeled)} appliance files, window={WINDOW} resolution)")
