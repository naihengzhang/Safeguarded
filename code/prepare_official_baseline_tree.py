from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def label(mask: Path) -> int:
    with Image.open(mask) as im:
        return int(im.convert("L").getbbox() is not None)


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.symlink(src.resolve(), dst)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2)
    a = p.parse_args()

    rows = {}
    for split in ("train", "test"):
        images = sorted(x for x in (a.data_root / split).glob("*.webp") if "(copy)" not in x.name)
        rows[split] = [(x, label(x.with_name(x.stem + "_GT.png"))) for x in images]
    normal = np.array([i for i, (_, y) in enumerate(rows["train"]) if y == 0])
    defect = np.array([i for i, (_, y) in enumerate(rows["train"]) if y == 1])
    rng = np.random.default_rng(a.seed)
    normal, defect = rng.permutation(normal), rng.permutation(defect)
    source, adapt_normal, adapt_defect = normal[:60], normal[92:156], defect[:20]

    variants = {
        "ksdd2_s2_clean": source,
        "ksdd2_s2_c20": np.concatenate([source, adapt_normal[:32], adapt_defect[:8]]),
    }
    manifest = {"seed": a.seed, "variants": {}}
    for name, indices in variants.items():
        base = a.output_root / name
        for i in indices:
            src, _ = rows["train"][int(i)]
            link(src, base / "train" / "good" / src.name)
        for src, y in rows["test"]:
            kind = "defect" if y else "good"
            link(src, base / "test" / kind / src.name)
            mask = src.with_name(src.stem + "_GT.png")
            if y:
                link(mask, base / "ground_truth" / kind / (src.stem + "_mask.png"))
        manifest["variants"][name] = {
            "train": len(indices), "hidden_train_defects": int(sum(rows["train"][int(i)][1] for i in indices)),
            "test": len(rows["test"]), "test_defects": int(sum(y for _, y in rows["test"])),
        }
    a.output_root.mkdir(parents=True, exist_ok=True)
    (a.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
