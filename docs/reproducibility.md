# Reproducibility record

This document maps the repository to the computational claims in the
manuscript and records known data limitations.

## Software and hardware

- Python: 3.10
- PyTorch: 2.5.1 (the package supports PyTorch 2.2 or newer)
- Ultralytics: 8.4.42
- Generalist training hardware: four NVIDIA GeForce RTX 5090 GPUs
- Mixed precision: enabled
- Input size: 512 x 512 for the paper benchmark
- Random seed in released configurations: 42
- AFSS sampling seed: 20260318

Exact driver and CUDA versions should be recorded in the archived release
after the final benchmark run. GPU kernels can introduce small numerical
differences even with deterministic algorithms enabled.

## Experiment-to-config mapping

| Manuscript experiment | Configuration change |
|---|---|
| Generalist model | `configs/paper_generalist.yaml` |
| Zero-shot | evaluate generalist checkpoint on new-domain test split |
| Full fine-tuning | `mode: full`, `adapter.enabled: false`, `afss.enabled: false` |
| Adapter | `mode: adapter`, `afss.enabled: false` |
| Adapter + AFSS | `configs/paper_adapter_afss.yaml` |

The generalist configuration reproduces Section 5.1: AdamW, initial learning
rate `1e-4`, total batch 32, 100 epochs, weight decay `1e-4`, five warm-up
epochs, cosine decay, and 512 x 512 inputs. The few-shot configuration uses
512 x 512 paper tiles together with the released implementation defaults for
interactive new-domain adaptation: 35 epochs, batch 8, initial learning rate
`1.5e-5`, and no online geometric or colour augmentation because augmentation
is applied during data preparation.

## Reported benchmark values

| Model | Precision | Recall | AP50 | Dice | BF1 |
|---|---:|---:|---:|---:|---:|
| YOLO26-seg-x | 87.9 | 88.1 | 85.5 | 68.1 | 62.7 |
| YOLO26-box-x + SAM | not tabulated | not tabulated | not tabulated | 86.5 | 82.1 |
| MineralSAM | **93.2** | **95.5** | **92.1** | **93.6** | **91.2** |

These are manuscript values. Reproduction requires the frozen split manifest
and released checkpoint; do not compare a new random split with this table.

## Availability limitations

The complete MineralInst pool cannot be mirrored in this Git repository because
some public sources restrict redistribution and some in-house images are
covered by sample or institutional agreements. The repository provides source
identifiers, formatting instructions, split rules, synthetic test data, and
executable training/evaluation code. Model weights are deliberately not
distributed. Until a permitted frozen split is available, the numerical tables
above are documented but are not independently reproducible from this
repository alone.

## Release checklist

- [ ] Add authors and final DOI to `CITATION.cff`.
- [ ] Add permitted real example inputs and expected outputs.
- [ ] Add the frozen paper split manifest with restricted paths anonymized.
- [ ] Run `pytest -q` in a clean environment.
- [ ] Run the synthetic smoke test from the README.
- [ ] Archive the exact tagged release in Zenodo or an equivalent repository.
