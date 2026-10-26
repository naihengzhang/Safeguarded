"""Create a compact, lossless 256x256 KSDD2 package.

The script preserves relative directories, split/metadata files and sample
basenames. RGB images become lossless WebP files; masks become PNG files and
are resized with nearest-neighbour interpolation. A manifest maps every source
file to its output file and records dimensions plus SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import shutil
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MASK_DIRECTORIES = {"mask", "masks", "masks_defect", "ground_truth", "gt", "label", "labels"}
MASK_SUFFIXES = ("_mask", "-mask", "_gt", "-gt", "_label", "-label")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_mask(relative_path: Path) -> bool:
    directory_match = any(part.lower() in MASK_DIRECTORIES for part in relative_path.parts[:-1])
    filename_match = relative_path.stem.lower().endswith(MASK_SUFFIXES)
    return directory_match or filename_match


def destination_path(output_root: Path, relative_path: Path, mask: bool) -> Path:
    suffix = ".png" if mask else ".webp"
    return output_root / relative_path.with_suffix(suffix)


def resize_one(source: Path, destination: Path, mask: bool, size: int) -> tuple[int, int, str]:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        original_size = image.size
        if mask:
            # Preserve integer class values; never interpolate mask labels.
            image = image.convert("L")
            image = ImageOps.pad(image, (size, size), method=Image.Resampling.NEAREST, color=0)
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, format="PNG", optimize=True)
        else:
            image = image.convert("RGB")
            image = ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR, color=(0, 0, 0))
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Lossless WebP reduces transfer size without introducing JPEG artifacts.
            image.save(destination, format="WEBP", lossless=True, method=6)
    return original_size[0], original_size[1], sha256(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize KSDD2 while preserving splits and labels.")
    parser.add_argument("input", type=Path, help="Extracted KSDD2 root containing train/ and test/.")
    parser.add_argument("output", type=Path, help="New output directory, for example KSDD2_256.")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--no-zip", action="store_true", help="Do not create OUTPUT.zip.")
    args = parser.parse_args()

    source_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if not source_root.is_dir():
        raise SystemExit(f"Input directory does not exist: {source_root}")
    if source_root == output_root or source_root in output_root.parents:
        raise SystemExit("Output must not be inside the input directory.")
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output directory must be absent or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    seen_destinations: set[Path] = set()
    image_count = mask_count = copied_count = 0

    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative = source.relative_to(source_root)
        # Windows can create accidental Explorer duplicates that are not part
        # of the official release. The public adapter excludes them explicitly.
        if "(copy)" in source.name.lower():
            continue
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            mask = is_mask(relative)
            destination = destination_path(output_root, relative, mask)
            if destination in seen_destinations:
                raise SystemExit(f"Output collision after extension conversion: {destination}")
            seen_destinations.add(destination)
            width, height, output_hash = resize_one(source, destination, mask, args.size)
            rows.append({
                "source_path": relative.as_posix(),
                "output_path": destination.relative_to(output_root).as_posix(),
                "kind": "mask" if mask else "image",
                "source_width": width,
                "source_height": height,
                "output_width": args.size,
                "output_height": args.size,
                "source_sha256": sha256(source),
                "output_sha256": output_hash,
            })
            mask_count += int(mask)
            image_count += int(not mask)
        else:
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied_count += 1

    manifest = output_root / "resize_manifest.csv"
    fieldnames = list(rows[0]) if rows else []
    if not fieldnames:
        raise SystemExit("No supported image files were found.")
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if not (output_root / "train").is_dir() or not (output_root / "test").is_dir():
        raise SystemExit("The output does not contain both train/ and test/; check the selected input root.")

    # KSDD2 contains one image and one mask per sample. Fail loudly if a
    # different tree was selected or if a file was silently lost.
    if (image_count, mask_count) != (3335, 3335):
        raise SystemExit(
            f"Expected 3335 images and 3335 masks after duplicate exclusion; got {image_count} and {mask_count}."
        )

    archive = None
    if not args.no_zip:
        archive = Path(shutil.make_archive(str(output_root), "zip", root_dir=output_root.parent, base_dir=output_root.name))

    print({
        "images": image_count,
        "masks": mask_count,
        "metadata_files_copied": copied_count,
        "manifest": str(manifest),
        "zip": str(archive) if archive else None,
    })


if __name__ == "__main__":
    main()
