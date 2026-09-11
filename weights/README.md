# Model weights

Model weights are not distributed in this repository and are excluded by
`.gitignore`. This keeps the source release small and avoids placing large
binary objects in Git history.

Expected local filenames are:

| File | Purpose |
|---|---|
| `mineralsam_generalist.pt` | Locally trained MineralPose YOLO26-seg-x generalist checkpoint |
| `sam_vit_b_01ec64.pth` | Optional official SAM ViT-B boundary-refinement checkpoint |

The paper configuration starts from the public `yolo26x-seg.pt` checkpoint and
can be used to train `mineralsam_generalist.pt`. The deterministic smoke test
starts from the smaller public `yolo26n-seg.pt`, which Ultralytics downloads
automatically.

Before using or archiving a locally trained checkpoint, record its SHA-256:

```bash
python -c "import hashlib,pathlib; p=pathlib.Path('weights/mineralsam_generalist.pt'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

Checkpoint absence is a reproducibility limitation: the exact numerical tables
in the manuscript cannot be reproduced without retraining on the frozen paper
split. The executable synthetic test case remains available without private
data or project-specific weights.

