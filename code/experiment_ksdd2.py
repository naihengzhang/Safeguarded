from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from core import Config, Gate, coreset, distances, safeguarded_scores, scores


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CACHE = ROOT / "data_cache"
PREDICTIONS = RESULTS / "ksdd2_predictions"
TRACES = RESULTS / "ksdd2_traces"
SEEDS = range(2, 7)
BRIGHTNESS = (1.0, 1.2)
CONTAMINATION = (0.0, 0.1, 0.2)
SAFE_MODES = ("unconditional", "rate_matched", "benefit", "local", "full")
CONFIG = Config()


def metrics(labels: np.ndarray, values: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        "AP": float(average_precision_score(labels, values)),
        "AUROC": float(roc_auc_score(labels, values)),
        "FPR": float(np.mean(values[labels == 0] > threshold)),
        "TPR": float(np.mean(values[labels == 1] > threshold)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def signflip(values: np.ndarray) -> float:
    permutations = np.array([np.mean(values * np.array(signs)) for signs in itertools.product([-1, 1], repeat=len(values))])
    return float(np.mean(permutations >= values.mean()))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    PREDICTIONS.mkdir(exist_ok=True)
    TRACES.mkdir(exist_ok=True)
    data = np.load(CACHE / "ksdd2_features.npz")
    train_labels = data["train_labels"]
    test_labels = data["test_labels"]
    train_ids = data["train_ids"]
    test_ids = data["test_ids"]
    normal_positions = data["normal_positions"]
    normal_to_probe = {int(position): i for i, position in enumerate(normal_positions)}
    normal_all = np.flatnonzero(train_labels == 0)
    defect_all = np.flatnonzero(train_labels == 1)
    tuning = json.loads((RESULTS / "tuning_lock.json").read_text())
    safeguard = json.loads((RESULTS / "safeguarded_lock.json").read_text())
    confidence_multiplier = float(tuning["confidence_multiplier"])
    rate_multiplier = float(tuning["rate_multiplier"])
    lam = float(safeguard["selected_lambda"])

    settings: dict[int, dict[str, object]] = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        normals = rng.permutation(normal_all)
        defects = rng.permutation(defect_all)
        split = {
            "source": normals[:60],
            "anchors": normals[60:68],
            "calibration": normals[68:92],
            "adapt_normal": normals[92:156],
            "adapt_defect": defects[:20],
        }
        source = coreset(data["train_1.0"][split["source"]].reshape(-1, 32), 128)
        anchors = data["train_1.0"][split["anchors"]]
        calibration = data["train_1.0"][split["calibration"]]
        sigma = float(max(np.median(distances(calibration, source)), 1e-6))
        threshold = float(np.quantile(scores(calibration, source), 0.95, method="higher"))
        probe_indices = np.array([normal_to_probe[int(i)] for i in split["anchors"]])
        nuisance = np.concatenate([data["train_1.2"][split["anchors"]]] * 2)
        probes = np.concatenate([data["train_scratch_normal"][probe_indices], data["train_patch_normal"][probe_indices]])
        support = np.zeros((16, 8, 8), dtype=bool)
        support[:, 2:6, 2:6] = True
        settings[seed] = {"split": split, "B": source, "A": anchors, "u": nuisance, "v": probes, "support": support.reshape(16, 64), "sigma": sigma, "threshold": threshold}

    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        setting = settings[seed]
        split = setting["split"]
        for brightness in BRIGHTNESS:
            train_x = data[f"train_{brightness}"]
            test_x = data[f"test_{brightness}"]
            for rate in CONTAMINATION:
                normal_count = 40 - round(40 * rate)
                stream = np.concatenate([split["adapt_normal"][:normal_count], split["adapt_defect"][:40 - normal_count]])
                stream = stream[np.random.default_rng(seed + 900).permutation(40)]
                memories: dict[str, tuple[np.ndarray, float, list[dict[str, object]]]] = {}
                for mode in ("unconditional", "confidence", "rate_matched", "benefit", "local", "full"):
                    gate_mode = "confidence" if mode in {"confidence", "rate_matched"} else mode
                    multiplier = confidence_multiplier if mode == "confidence" else rate_multiplier if mode == "rate_matched" else 1.0
                    gate = Gate(setting["B"], setting["A"], setting["u"], setting["v"], setting["support"], setting["sigma"], setting["threshold"], gate_mode, CONFIG, "high", seed, setting["threshold"] * multiplier)
                    trace = []
                    start = time.perf_counter()
                    for index in stream:
                        trace.append(gate.step(train_x[index], str(train_ids[index])))
                    elapsed = time.perf_counter() - start
                    memories[mode] = (gate.memory.copy(), elapsed, trace)
                    tag = f"s{seed}_b{brightness}_c{rate}_{mode}"
                    with (TRACES / f"{tag}.jsonl").open("w") as stream_file:
                        for item in trace:
                            stream_file.write(json.dumps({k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in item.items()}) + "\n")

                frozen = scores(test_x, setting["B"])
                method_scores = {"frozen": frozen}
                for mode in SAFE_MODES:
                    method_scores[f"{mode}_safe"] = safeguarded_scores(test_x, setting["B"], memories[mode][0], setting["threshold"], lam)
                method_scores["confidence_safe"] = safeguarded_scores(test_x, setting["B"], memories["confidence"][0], setting["threshold"], lam)
                method_scores["direct_full"] = scores(test_x, memories["full"][0])

                for method, values in method_scores.items():
                    base_mode = method.removesuffix("_safe") if method.endswith("_safe") else "full" if method == "direct_full" else None
                    trace = memories[base_mode][2] if base_mode in memories else []
                    acceptance = float(np.mean([item["accepted"] for item in trace])) if trace else 0.0
                    decision_violations = int(np.sum((values > setting["threshold"]) != (frozen > setting["threshold"]))) if method.endswith("_safe") else 0
                    result = {
                        "dataset": "KSDD2",
                        "seed": seed,
                        "brightness": brightness,
                        "contamination": rate,
                        "method": method,
                        "acceptance": acceptance,
                        "decision_violations": decision_violations,
                        **metrics(test_labels, values, setting["threshold"]),
                    }
                    rows.append(result)
                tag = f"s{seed}_b{brightness}_c{rate}"
                np.savez_compressed(PREDICTIONS / f"{tag}.npz", ids=test_ids, labels=test_labels, threshold=setting["threshold"], **method_scores)
                print("KSDD2", tag, "complete", flush=True)

    write_csv(RESULTS / "ksdd2_confirmation.csv", rows)
    methods = sorted(set(row["method"] for row in rows))
    summary: dict[str, object] = {
        "dataset": "KSDD2",
        "official_train_test_split": True,
        "parameter_source": "VisA development categories only; no KSDD2 label-based tuning",
        "settings": len(SEEDS) * len(BRIGHTNESS) * len(CONTAMINATION),
        "test_images": len(test_labels),
        "test_normal": int(np.sum(test_labels == 0)),
        "test_defect": int(np.sum(test_labels == 1)),
        "lambda": lam,
        "methods": {},
    }
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        summary["methods"][method] = {metric: float(np.mean([row[metric] for row in subset])) for metric in ["AP", "AUROC", "FPR", "TPR", "acceptance"]}

    full_by_seed = []
    for seed in SEEDS:
        proposed = np.mean([row["AP"] for row in rows if row["seed"] == seed and row["method"] == "full_safe"])
        frozen_ap = np.mean([row["AP"] for row in rows if row["seed"] == seed and row["method"] == "frozen"])
        full_by_seed.append(float(proposed - frozen_ap))
    values = np.array(full_by_seed)
    rng = np.random.default_rng(20260923)
    bootstrap = np.mean(rng.choice(values, (100000, len(values)), replace=True), axis=1)
    summary["full_safe_minus_frozen"] = {
        "seed_deltas": full_by_seed,
        "mean": float(values.mean()),
        "seed_bootstrap_95_CI": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "one_sided_exact_seed_signflip_p": signflip(values),
        "caution": "Seeds vary stream construction but share the official test identities; this is a blocked robustness analysis, not five independent datasets.",
    }
    (RESULTS / "ksdd2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
