"""MineralPose inference with optional SAM mask-prompt refinement."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class SAMMaskRefiner:
    def __init__(self, checkpoint: str | Path, model_type: str = "vit_b", device: str = "cuda"):
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as error:
            raise ImportError("Install SAM support with `python -m pip install -e .[sam]`") from error
        model = sam_model_registry[model_type](checkpoint=str(checkpoint))
        model.to(device=device)
        self.predictor = SamPredictor(model)

    def set_image(self, image_rgb: np.ndarray) -> None:
        self.predictor.set_image(image_rgb)

    def refine(self, coarse_mask: np.ndarray) -> np.ndarray:
        low_resolution = cv2.resize(
            coarse_mask.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR
        )
        mask_logits = np.where(low_resolution >= 0.5, 8.0, -8.0).astype(np.float32)[None]
        masks, scores, _ = self.predictor.predict(
            mask_input=mask_logits,
            multimask_output=True,
            return_logits=False,
        )
        return masks[int(np.argmax(scores))].astype(bool)


def _colour(instance_id: int) -> np.ndarray:
    generator = np.random.default_rng(instance_id * 7919)
    return generator.integers(48, 240, size=3, dtype=np.uint8)


def _overlay(image_rgb: np.ndarray, instance_map: np.ndarray) -> np.ndarray:
    coloured = image_rgb.copy()
    for instance_id in np.unique(instance_map):
        if instance_id == 0:
            continue
        mask = instance_map == instance_id
        coloured[mask] = (0.45 * coloured[mask] + 0.55 * _colour(int(instance_id))).astype(np.uint8)
    return coloured


def predict(
    weights: str | Path,
    source: str | Path,
    output: str | Path,
    *,
    confidence: float = 0.25,
    image_size: int = 512,
    device: str = "",
    max_detections: int = 3000,
    sam_checkpoint: str | Path | None = None,
    sam_type: str = "vit_b",
) -> list[dict]:
    from ultralytics import YOLO

    output = Path(output)
    instance_dir = output / "instances"
    overlay_dir = output / "overlays"
    instance_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    refiner = SAMMaskRefiner(sam_checkpoint, sam_type, device or "cuda") if sam_checkpoint else None
    model = YOLO(str(weights))
    records = []
    results = model.predict(
        source=str(source),
        conf=confidence,
        imgsz=image_size,
        device=device or None,
        max_det=max_detections,
        retina_masks=True,
        stream=True,
        verbose=False,
    )
    for result in results:
        image_bgr = np.asarray(result.orig_img)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        instance_map = np.zeros((height, width), dtype=np.uint16)
        masks = np.empty((0, height, width), dtype=bool)
        confidences = np.empty(0, dtype=float)
        if result.masks is not None:
            masks = result.masks.data.detach().cpu().numpy() >= 0.5
            confidences = result.boxes.conf.detach().cpu().numpy()
        if refiner is not None and len(masks):
            refiner.set_image(image_rgb)
        order = np.argsort(confidences)
        for instance_id, index in enumerate(order, start=1):
            mask = masks[index]
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            if refiner is not None:
                mask = refiner.refine(mask)
            instance_map[mask] = instance_id
        stem = Path(result.path).stem
        cv2.imwrite(str(instance_dir / f"{stem}.png"), instance_map)
        cv2.imwrite(
            str(overlay_dir / f"{stem}.png"),
            cv2.cvtColor(_overlay(image_rgb, instance_map), cv2.COLOR_RGB2BGR),
        )
        records.append(
            {
                "image": Path(result.path).name,
                "height": height,
                "width": width,
                "instances": int(instance_map.max()),
                "confidences": [round(float(value), 6) for value in confidences],
                "sam_refined": refiner is not None,
            }
        )
    (output / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records

