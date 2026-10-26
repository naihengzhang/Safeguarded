# Dataset placement

No dataset is redistributed in this repository.

## Required for the reported experiments: VisA

1. Review the dataset license and instructions in the official Amazon Science project:
   https://github.com/amazon-science/spot-diff
2. Official public dataset record:
   https://registry.opendata.aws/visa/
3. Download the official archive:
   https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
4. Place it at:

```text
data/VisA_20220922.tar
```

The archive is intentionally ignored by Git. Feature caches are generated under `data_cache/`, which is also ignored.

## Required for the reported external validation: KSDD2

1. Review the official page, license, counts, and citation:
   https://www.vicos.si/resources/kolektorsdd2/
2. Download the official archive:
   https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip
3. Extract it to `data/KolektorSDD2/`.
4. Create the compact 256 by 256 copy while retaining `train/`, `test/`, masks, and basenames:

```text
python3 code/resize_ksdd2.py data/KolektorSDD2 data/KolektorSDD2_256 --no-zip
```

The adapter asserts 3335 image/mask pairs and excludes accidental `(copy)` filenames. KSDD2 is licensed CC BY-NC-SA 4.0; dataset files are intentionally ignored by Git.
