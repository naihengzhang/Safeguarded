# Decision-Safeguarded Adaptive Memory Scoring

Reproducible code and final aggregate results for the accompanying manuscript. The repository executes a compact ResNet-18 memory detector on VisA and KSDD2, applies a prospective admission gate, and preserves every frozen-detector decision at the fixed source threshold through decision-safeguarded residual scoring.

## Reported evidence

- VisA locked confirmation: AP 0.84319 versus 0.83566 for frozen scoring; category-macro difference +0.752 percentage points, 95% category-bootstrap interval [0.410, 1.120], Holm-adjusted exact sign-flip p=0.0273.
- KSDD2 locked external transfer: AP 0.76055 versus 0.75906 for frozen scoring; seed-blocked difference +0.149 points, interval [-0.150, 0.534], p=0.3125. This direction is positive but inconclusive, and rate-matched safeguarding is numerically best.
- Fixed-threshold decision violations: zero on both datasets.

These are matched mechanism-study results, not official leaderboard claims. The full gate is not claimed to outperform every simpler safeguarded gate.

### Official-code KSDD2 contextual baselines

Using seed 2, the same 60 clean source images, all 1004 official test images, WideResNet-50 layer2/layer3 features and a 1% coreset, official PatchCore obtains AP/AUROC 0.85445/0.94784. With a 100-image training tree containing 8 hidden defects, it obtains 0.85478/0.94753. Official SoftPatch obtains 0.84845/0.93805 and 0.85078/0.94717, respectively. These single-seed anchors exceed the compact ResNet-18 proposed instantiation in absolute accuracy; the manuscript therefore makes no SOTA-accuracy claim. Exact outputs are in `results/official_baselines_ksdd2.csv`.

## Data: download it yourself

No images, masks, pretrained weights, feature caches, per-run predictions, or traces are redistributed.

### VisA (CC BY 4.0)

- Official project: https://github.com/amazon-science/spot-diff
- Public record: https://registry.opendata.aws/visa/
- Official archive: https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar

Place the archive at `data/VisA_20220922.tar`.

### KSDD2 (CC BY-NC-SA 4.0)

- Official page, license, counts, and citation: https://www.vicos.si/resources/kolektorsdd2/
- Official archive: https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip

Extract the official archive, then create the compact split-preserving copy:

```bash
python3 code/resize_ksdd2.py data/KolektorSDD2 data/KolektorSDD2_256 --no-zip
```

The converter uses bilinear aspect-preserving padding for RGB images, nearest-neighbour padding for masks, excludes accidental filenames containing `(copy)`, and asserts the official 3335 image/mask pairs. Place the result at `data/KolektorSDD2_256/`.

## Environment

Python 3.12 was used. Install the CPU build of PyTorch first, then the remaining pinned dependencies:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu torchvision==0.29.0+cpu
python -m pip install -r requirements.txt
```

No GPU is required. Full feature extraction is the slowest stage.

## Reproduction

Run from the repository root:

```bash
python3 code/run_pipeline.py test
python3 code/run_pipeline.py prepare \
  --archive data/VisA_20220922.tar \
  --ksdd2-root data/KolektorSDD2_256
python3 code/run_pipeline.py experiment
python3 code/run_pipeline.py audit
python3 code/run_pipeline.py figures
```

To rerun only the external dataset:

```bash
python3 code/prepare_ksdd2.py --data-root data/KolektorSDD2_256
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 code/experiment_ksdd2.py
python3 code/audit_ksdd2.py
```

To reconstruct the matched MVTec-style trees used by the official PatchCore and SoftPatch repositories:

```bash
python3 code/prepare_official_baseline_tree.py \
  --data-root data/KolektorSDD2_256 \
  --output-root data/official_baseline_tree \
  --seed 2
```

The official repositories are not vendored. The recorded revisions are PatchCore `fcaa92f124fb1ad74a7acf56726decd4b27cbcad` and SoftPatch `f8dc3e0b2e70bc3b5a606cc414c7703c6c5e7810`. Reproduction requires cloning those repositories, adding image-level AP reporting, and selecting CPU when no CUDA device is available. The manuscript documents the exact detector arguments; `results/official_baselines_ksdd2.csv` is the immutable final output.

## Repository policy

Only source code, locked configurations, tests, final aggregate results, audit summaries, and final vector figures are versioned. Regenerable caches and raw/intermediate outputs are excluded by `.gitignore`. See `data/README.md` for dataset placement and licensing reminders.


## Citation and archival DOI

A Zenodo DOI has not yet been minted. After creating the public GitHub release, archive that release in Zenodo and replace this paragraph with the DOI badge and manuscript citation. Do not cite a placeholder DOI.
