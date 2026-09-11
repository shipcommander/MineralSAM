# MineralSAM

Reference implementation for **MineralSAM: A multi-domain generalist model for
class-agnostic grain instance segmentation and quantitative petrographic
analysis of rock thin sections**.

MineralSAM uses a YOLO26 segmentation model (MineralPose) to generate a coarse
mask for each grain and passes each mask to SAM as a spatial prompt for boundary
refinement. For a new basin, lithology, preparation procedure, or microscope,
the repository provides a lightweight parallel convolutional Adapter and AFSS
sample scheduling for parameter-efficient few-shot adaptation.

> Terminology: the scheduling method in the manuscript is **AFSS**, not ASFF.
> AFSS changes which images are sampled during training; it is unrelated to
> adaptive spatial feature fusion.

## Main results reported in the manuscript

| Setting | New-domain AP50 | Source-domain AP50 | Source drop |
|---|---:|---:|---:|
| Zero-shot | 74.6 | 92.1 | 0.0 pp |
| Full fine-tuning (10 fields) | 89.2 | 83.8 | 8.3 pp |
| Adapter (10 fields) | 88.7 | 90.5 | 1.6 pp |
| Adapter + AFSS (10 fields) | **90.1** | **91.3** | **0.8 pp** |

On the in-distribution test set, the reported Precision, Recall, AP50, Dice,
and BF1 are 93.2%, 95.5%, 92.1%, 93.6%, and 91.2%, respectively. These values
are reference results, not claims about the synthetic smoke-test data.

## Repository contents

```text
configs/                 Exact paper and small demo configurations
data/                    Dataset format, source list, and split-manifest schema
docs/                    User guide and reproducibility notes
mineralsam/              Adapter, AFSS, training, inference, and metrics
scripts/                 Synthetic demo-data generator
tests/                   Unit tests for the new computational components
weights/                 Checkpoint download and license notes
```

## Installation

Python 3.10 or 3.11 is recommended. Install PyTorch for the CUDA version on
your machine, then install this repository:

```bash
conda env create -f environment.yml
conda activate mineralsam
```

For a CPU-only smoke test:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

SAM refinement is optional because its checkpoint is large:

```bash
python -m pip install -e ".[sam]"
```

## Five-minute smoke test

Generate a deterministic synthetic instance-segmentation dataset and run one
small epoch. The demo verifies data loading, Adapter insertion, AFSS setup, and
checkpoint writing; it is not a scientific benchmark.

```bash
python scripts/make_demo_data.py
mineralsam train --config configs/demo_adapter_afss.yaml
pytest -q
```

Ultralytics downloads `yolo26n-seg.pt` automatically when it is not present.
The demo can be run on CPU, although a CUDA GPU is recommended.

## Paper workflow

1. Prepare class-agnostic YOLO segmentation labels as described in
   [data/README.md](data/README.md).
2. Edit the dataset and checkpoint paths in a configuration file.
3. Train the multi-source generalist model:

   ```bash
   mineralsam train --config configs/paper_generalist.yaml
   ```

4. Adapt to approximately ten annotated fields from a new domain:

   ```bash
   mineralsam train --config configs/paper_adapter_afss.yaml
   ```

5. Generate grain-instance maps, optionally with SAM refinement:

   ```bash
   mineralsam predict --weights runs/adapter_afss/weights/best.pt \
     --source path/to/images --output runs/predictions

   mineralsam predict --weights runs/adapter_afss/weights/best.pt \
     --source path/to/images --output runs/refined \
     --sam-checkpoint weights/sam_vit_b_01ec64.pth --sam-type vit_b
   ```

6. Evaluate predicted and reference instance-ID maps:

   ```bash
   mineralsam evaluate --pred runs/refined/instances \
     --target path/to/reference_instance_maps --output runs/metrics.json
   ```

See [docs/user_guide.md](docs/user_guide.md) for inputs and outputs and
[docs/reproducibility.md](docs/reproducibility.md) for the paper settings.

## Adapter and AFSS in one paragraph

For each eligible frozen C3k2 block, the Adapter learns a residual correction
`Y = F(X) + A(X)`. A 1 x 1 projection creates a bottleneck, one or two
pointwise residual blocks process part of the channels, concatenated features
are projected back, and a learnable residual scale is initialized to 0.25. The
final projection is initialized with standard deviation `1e-3`, so adaptation
starts close to the pretrained model. AFSS scores every training image with
`min(P_box, R_box, P_mask, R_mask)`: hard images are always used, moderate
images are sampled at 40%, and easy images at 2%, with periodic review to limit
forgetting.

## Data and checkpoints

MineralInst combines permitted samples from public resources with in-house
thin-section images. Some source licenses and institutional agreements prevent
redistribution of the complete image pool. This repository therefore includes
source identifiers, preprocessing and split rules, and a deterministic
synthetic test case. Model weights are not uploaded; this limitation and the
local checkpoint locations are documented in [weights/README.md](weights/README.md).

## License and third-party software

The original code in this repository is released under the MIT License.
Ultralytics and Segment Anything are separate third-party projects with their
own licenses. In particular, Ultralytics 8.4.42 is distributed under AGPL-3.0;
users are responsible for complying with all third-party terms.

## Citation

The manuscript is under review. Please use [CITATION.cff](CITATION.cff); the
final bibliographic record and DOI will be added after publication.
