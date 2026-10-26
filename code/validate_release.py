from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
required = [
    "README.md", "LICENSE", "requirements.txt", "data/README.md",
    "code/core.py", "code/prepare.py", "code/prepare_ksdd2.py", "code/resize_ksdd2.py",
    "code/experiment_ksdd2.py", "code/audit_ksdd2.py", "code/run_pipeline.py", "code/test_core.py",
    "results/matched_baseline_summary.json", "results/safeguarded_summary.json",
    "results/audit.json", "results/safeguarded_audit.json",
    "results/ksdd2_summary.json", "results/ksdd2_audit.json",
]
missing = [p for p in required if not (ROOT / p).is_file()]
forbidden = ["data_cache", "results/predictions", "results/traces", "results/splits", "results/ksdd2_predictions", "results/ksdd2_traces", "__pycache__"]
present_forbidden = [p for p in forbidden if (ROOT / p).exists()]
summary = json.loads((ROOT / "results" / "matched_baseline_summary.json").read_text())
assert abs(summary["proposed"]["AP"] - 0.843185012664024) < 1e-12
assert abs(summary["category_paired_comparisons"]["frozen"]["p_holm"] - 0.02734375) < 1e-12
external = json.loads((ROOT / "results" / "ksdd2_summary.json").read_text())
assert external["settings"] == 30 and external["test_images"] == 1004
assert abs(external["methods"]["full_safe"]["AP"] - 0.7605494624301044) < 1e-12
if missing or present_forbidden:
    raise SystemExit({"missing": missing, "forbidden": present_forbidden})
print({"release_valid": True, "required_files": len(required), "forbidden_artifacts": 0})
