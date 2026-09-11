# Data format and availability

## Directory layout

MineralSAM uses the Ultralytics polygon-segmentation format:

```text
dataset_root/
  images/
    train/ or train.txt
    val/   or val.txt
    test/  or test.txt
  labels/
    train/
    val/
    test/
  dataset.yaml
  split_manifest.csv
```

Each RGB image has a same-stem `.txt` label. One grain instance is stored per
line as `class_id x1 y1 x2 y2 ...`, where polygon coordinates are normalized to
`[0, 1]`. MineralSAM is class-agnostic, so the released experiments use class
ID `0` (`grain`) for every instance.

## Split rule

Split by the original thin section and sample source **before** creating 512 x
512 tiles. Tiles from one physical thin section must never occur in more than
one of train, validation, and test. Record every tile in a CSV manifest with:

```text
tile_id,source_dataset,country_or_basin,lithology,modality,slide_id,split
```

The test annotations used for the manuscript were independently reviewed by
petrographers. Do not tune thresholds on the test split.

## MineralInst provenance

MineralInst contains selected, quality-controlled images from:

- Science Data Bank (ScienceDB): <https://www.scidb.cn/en/>
- MUMDMC2025: DOI `10.1038/s41597-025-05879-9`
- DeepCarbonate: DOI `10.1038/s41597-026-06633-5`
- LITHOS: arXiv `2511.00328`
- In-house thin sections from multiple basins and imaging systems

The source publications describe larger resources; MineralInst does not claim
to redistribute every source image. Check the original license before using or
redistributing a source. In-house images cannot be redistributed where sample
or institutional agreements prohibit it.

## Preprocessing

1. Remove blank, severely defocused, strongly unevenly illuminated, and
   duplicate fields.
2. Normalize scale metadata and retain pixel-to-length calibration separately.
3. Split by `slide_id` and source.
4. Tile into 512 x 512 images; retain the parent slide and tile coordinates in
   the manifest.
5. Convert reviewed instance contours to normalized YOLO polygons.
6. Run label checks for polygon validity, image-label pairing, and split
   leakage.

The manuscript reports 234,713 valid tiles and approximately 12.3 million
reviewed instances. The current generalist model was trained on 30% of the
complete pool.

## Synthetic test data

Run `python scripts/make_demo_data.py`. It creates deterministic colored grain
fields and exact polygon labels under `data/demo/`. The generated images are
CC0 test fixtures and are not part of MineralInst or any reported experiment.

