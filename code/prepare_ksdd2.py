from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data_cache"
RESULTS = ROOT / "results"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mask_label(mask: Path) -> int:
    with Image.open(mask) as image:
        return int(image.convert("L").getbbox() is not None)


def image_array(path: Path, brightness: float = 1.0) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if brightness != 1.0:
            rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
        return np.asarray(rgb, dtype=np.uint8).copy()


def records(data_root: Path, split: str) -> list[dict[str, object]]:
    folder = data_root / split
    images = sorted(p for p in folder.glob("*.webp") if "(copy)" not in p.name)
    out = []
    for image in images:
        mask = folder / f"{image.stem}_GT.png"
        if not mask.is_file():
            raise FileNotFoundError(f"Missing mask for {image}")
        out.append({
            "id": image.stem,
            "split": split,
            "image": str(image.relative_to(data_root)),
            "mask": str(mask.relative_to(data_root)),
            "label": mask_label(mask),
            "image_sha256": file_hash(image),
            "mask_sha256": file_hash(mask),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic KSDD2 feature caches.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing train/ and test/.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    output = CACHE / "ksdd2_features.npz"
    if output.exists() and not args.force:
        print(f"Using existing cache: {output}")
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    train = records(data_root, "train")
    test = records(data_root, "test")
    if (len(train), sum(r["label"] for r in train), len(test), sum(r["label"] for r in test)) != (2331, 246, 1004, 110):
        raise ValueError("KSDD2 counts do not match the official split after duplicate removal.")

    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    projection_path = RESULTS / "projection.npy"
    if projection_path.is_file():
        projection = np.load(projection_path).astype("float32")
    else:
        projection = np.random.default_rng(2026).normal(size=(384, 32)).astype("float32") / np.sqrt(32)
        np.save(projection_path, projection)
    if projection.shape != (384, 32):
        raise ValueError(f"Unexpected projection shape: {projection.shape}")

    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

    def extract(images: list[np.ndarray]) -> np.ndarray:
        batches = []
        with torch.inference_mode():
            for start in range(0, len(images), args.batch_size):
                array = np.stack(images[start:start + args.batch_size])
                x = torch.from_numpy(array.transpose(0, 3, 1, 2).copy()).float() / 255
                x = (x - mean) / std
                h = model.maxpool(model.relu(model.bn1(model.conv1(x))))
                h = model.layer1(h)
                layer2 = model.layer2(h)
                layer3 = model.layer3(layer2)
                h = torch.cat([F.adaptive_avg_pool2d(layer2, (8, 8)), F.adaptive_avg_pool2d(layer3, (8, 8))], 1)
                z = h.permute(0, 2, 3, 1).reshape(len(array), 64, 384).numpy() @ projection
                z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
                batches.append(z.astype("float32"))
                print(f"features {min(start + args.batch_size, len(images))}/{len(images)}", flush=True)
        return np.concatenate(batches)

    start = time.perf_counter()
    arrays: dict[str, np.ndarray] = {}
    for split, rows in (("train", train), ("test", test)):
        paths = [data_root / str(row["image"]) for row in rows]
        for brightness in (1.0, 1.2):
            arrays[f"{split}_{brightness}"] = extract([image_array(path, brightness) for path in paths])

    normal_positions = np.array([i for i, row in enumerate(train) if row["label"] == 0], dtype=np.int32)
    normal_paths = [data_root / str(train[i]["image"]) for i in normal_positions]
    for family in ("scratch", "patch"):
        altered = []
        for path in normal_paths:
            image = image_array(path, 1.2)
            if family == "scratch":
                image[108:148, 124:132] = 255
            else:
                image[112:144, 112:144] = np.indices((32, 32)).sum(0)[..., None] % 2 * 255
            altered.append(image)
        arrays[f"train_{family}_normal"] = extract(altered)

    arrays["train_labels"] = np.array([r["label"] for r in train], dtype=np.int8)
    arrays["test_labels"] = np.array([r["label"] for r in test], dtype=np.int8)
    arrays["train_ids"] = np.array([r["id"] for r in train])
    arrays["test_ids"] = np.array([r["id"] for r in test])
    arrays["normal_positions"] = normal_positions
    np.savez_compressed(output, **arrays)

    model_hash = hashlib.sha256(b"".join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    manifest = {
        "dataset": "KolektorSDD2",
        "source": "https://www.vicos.si/resources/kolektorsdd2/",
        "license": "CC BY-NC-SA 4.0",
        "duplicate_exclusions": ["train/10301 (copy).webp", "train/10301_GT (copy).webp"],
        "train_images": len(train),
        "train_normal": int(sum(not r["label"] for r in train)),
        "train_defect": int(sum(r["label"] for r in train)),
        "test_images": len(test),
        "test_normal": int(sum(not r["label"] for r in test)),
        "test_defect": int(sum(r["label"] for r in test)),
        "records": train + test,
        "encoder": "torchvision ResNet18 IMAGENET1K_V1",
        "encoder_sha256": model_hash,
        "projection_sha256": file_hash(projection_path),
        "seconds": time.perf_counter() - start,
    }
    (RESULTS / "ksdd2_data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ["train_images", "train_normal", "train_defect", "test_images", "test_normal", "test_defect", "seconds"]}, indent=2))


if __name__ == "__main__":
    main()
