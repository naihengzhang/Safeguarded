from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    report = pd.read_csv(RESULTS / "ksdd2_confirmation.csv")
    manifest = json.loads((RESULTS / "ksdd2_data_manifest.json").read_text())
    assert (manifest["train_images"], manifest["train_normal"], manifest["train_defect"]) == (2331, 2085, 246)
    assert (manifest["test_images"], manifest["test_normal"], manifest["test_defect"]) == (1004, 894, 110)
    assert len(manifest["records"]) == 3335
    train_ids = {r["id"] for r in manifest["records"] if r["split"] == "train"}
    test_ids = {r["id"] for r in manifest["records"] if r["split"] == "test"}
    assert not (train_ids & test_ids)

    checked = 0
    max_error = 0.0
    max_violations = 0
    for prediction in sorted((RESULTS / "ksdd2_predictions").glob("*.npz")):
        data = np.load(prediction)
        labels = data["labels"]
        frozen = data["frozen"]
        threshold = float(data["threshold"])
        seed, brightness, contamination = prediction.stem.split("_")
        key = (
            int(seed.removeprefix("s")),
            float(brightness.removeprefix("b")),
            float(contamination.removeprefix("c")),
        )
        subset = report[
            (report.seed == key[0])
            & (report.brightness == key[1])
            & (report.contamination == key[2])
        ]
        assert len(subset) == 8
        for _, row in subset.iterrows():
            values = data[row.method]
            actual = {
                "AP": average_precision_score(labels, values),
                "AUROC": roc_auc_score(labels, values),
                "FPR": np.mean(values[labels == 0] > threshold),
                "TPR": np.mean(values[labels == 1] > threshold),
            }
            max_error = max(max_error, *(abs(float(row[k]) - float(v)) for k, v in actual.items()))
            if row.method.endswith("_safe"):
                violations = int(np.sum((values > threshold) != (frozen > threshold)))
                max_violations = max(max_violations, violations)
                assert violations == int(row.decision_violations) == 0
            checked += 1
    assert checked == 240
    assert max_error < 1e-12
    out = {
        "dataset": "KSDD2",
        "prediction_files": 30,
        "method_setting_rows": checked,
        "maximum_metric_recomputation_error": max_error,
        "maximum_safeguarded_decision_violations": max_violations,
        "train_test_id_overlap": 0,
        "official_split_counts_verified": True,
    }
    (RESULTS / "ksdd2_audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
