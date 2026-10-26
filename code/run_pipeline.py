from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

STAGES = {
    "test": ["test_core.py"],
    "prepare": ["prepare.py"],
    "experiment": ["experiment.py", "safeguarded_experiment.py", "baseline_confirmation.py", "complete_proposer_sensitivity.py", "reanalyze_confirmation.py"],
    "audit": ["audit.py", "audit_safeguarded.py"],
    "figures": ["plots.py", "plot_matched_confirmation.py"],
}


def run(script: str, extra: list[str] | None = None) -> None:
    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    cmd = [sys.executable, str(ROOT / "code" / script), *(extra or [])]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the reported VisA and KSDD2 study.")
    parser.add_argument("stage", choices=["test", "prepare", "experiment", "audit", "figures", "all"])
    parser.add_argument("--archive", type=Path, default=ROOT / "data" / "VisA_20220922.tar")
    parser.add_argument("--ksdd2-root", type=Path, default=ROOT / "data" / "KolektorSDD2_256")
    args = parser.parse_args()
    order = ["test", "prepare", "experiment", "audit", "figures"] if args.stage == "all" else [args.stage]
    for stage in order:
        for script in STAGES[stage]:
            run(script, ["--archive", str(args.archive)] if script == "prepare.py" else None)
        if stage == "prepare":
            run("prepare_ksdd2.py", ["--data-root", str(args.ksdd2_root)])
        elif stage == "experiment":
            run("experiment_ksdd2.py")
        elif stage == "audit":
            run("audit_ksdd2.py")


if __name__ == "__main__":
    main()
