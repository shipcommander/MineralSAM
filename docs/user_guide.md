# User guide

## Inputs

- **Training images:** RGB PNG/JPEG thin-section tiles.
- **Training labels:** normalized YOLO polygon instance labels; use class `0`
  for every grain.
- **Dataset YAML:** Ultralytics dataset description with train, validation, and
  test paths.
- **Weights:** a YOLO26 segmentation checkpoint. For Adapter training, start
  from the MineralSAM generalist checkpoint.
- **Optional SAM checkpoint:** ViT-B, ViT-L, or ViT-H checkpoint compatible
  with Meta's Segment Anything implementation.

PPL and XPL images are treated as independent samples. Registered PPL-XPL pairs
are not required by the class-agnostic segmentation model.

## Training modes

`mode: generalist` trains the complete MineralPose network. `mode: full` is the
full fine-tuning baseline on a new domain. `mode: adapter` freezes the host
network, inserts parallel convolutional Adapters into eligible C3k2 blocks,
and optimizes Adapter parameters (plus the segmentation head only when
`train_segment_head: true`).

AFSS is independent of the parameter-training mode and is enabled under the
`afss` configuration block. Disable it with `enabled: false` for the Adapter
ablation.

```bash
mineralsam train --config configs/paper_adapter_afss.yaml \
  --weights /path/to/generalist.pt --data /path/to/dataset.yaml
```

The run directory contains Ultralytics logs and checkpoints plus
`afss_state.json`, which stores per-image sufficiency and last-use epochs.

## AFSS behaviour

After the full-data warm-up, every image is evaluated at the configured update
interval. Its sufficiency is:

```text
S_i = min(P_box, R_box, P_mask, R_mask)
```

- Easy: `S_i > 0.85`; sample about 2%, revisit after ten unused epochs.
- Moderate: `0.55 <= S_i <= 0.85`; sample about 40%, force a revisit after
  three unused epochs.
- Hard: `S_i < 0.55`; use every epoch.

The default update interval is five epochs. State is written atomically and can
be reused after an interrupted run.

## Prediction

Without `--sam-checkpoint`, the command saves coarse MineralPose instance maps.
With it, every coarse mask is resized to SAM's 256 x 256 mask-prompt space and
used to obtain a refined boundary.

```bash
mineralsam predict --weights weights/mineralsam_generalist.pt \
  --source examples --output runs/predictions --confidence 0.25
```

Outputs:

- `instances/<stem>.png`: 16-bit instance-ID map (`0` is background).
- `overlays/<stem>.png`: color overlay for visual quality control.
- `predictions.json`: image names, shapes, instance counts, and confidences.

For whole-slide mosaics, tile externally with overlap, discard low-confidence
edge fragments, and merge instances across overlaps before computing
petrographic descriptors. Pixel thresholds must be rescaled with microscope
resolution.

## Evaluation

Reference and prediction maps must have identical shapes. Each nonzero integer
is an independent instance. The evaluator performs one-to-one IoU matching,
then reports instance Precision and Recall, mean Dice, and mean boundary F1.

```bash
mineralsam evaluate --pred runs/predictions/instances \
  --target data/reference_instances --boundary-tolerance 2
```

Ultralytics AP50 and mAP50-95 are produced by validation during training. Use
the same confidence, IoU, maximum-detection, and image-size settings for all
model comparisons.

## Expected behaviour and limitations

MineralSAM segments grain/particle instances; it does not assign mineral names.
SAM refinement improves local contours but cannot recover a grain that
MineralPose completely misses. Extremely dense fine grains may require a
higher `max_det`. Grain-contact and physical-size measurements require a valid
pixel scale and careful treatment of objects clipped at tile borders.

