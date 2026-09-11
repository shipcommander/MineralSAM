"""Instance-map metrics used for region and boundary quality checks."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def _instances(label_map: np.ndarray) -> list[np.ndarray]:
    return [label_map == value for value in np.unique(label_map) if value != 0]


def _iou_matrix(predicted: list[np.ndarray], target: list[np.ndarray]) -> np.ndarray:
    matrix = np.zeros((len(predicted), len(target)), dtype=np.float64)
    for row, prediction in enumerate(predicted):
        for column, reference in enumerate(target):
            intersection = np.count_nonzero(prediction & reference)
            union = np.count_nonzero(prediction | reference)
            matrix[row, column] = intersection / union if union else 0.0
    return matrix


def _boundary(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    return mask & ~eroded


def boundary_f1(prediction: np.ndarray, target: np.ndarray, tolerance: int = 2) -> float:
    pred_boundary, target_boundary = _boundary(prediction), _boundary(target)
    kernel_size = 2 * max(0, int(tolerance)) + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    pred_neighbourhood = cv2.dilate(pred_boundary.astype(np.uint8), kernel) > 0
    target_neighbourhood = cv2.dilate(target_boundary.astype(np.uint8), kernel) > 0
    precision = np.count_nonzero(pred_boundary & target_neighbourhood) / max(1, np.count_nonzero(pred_boundary))
    recall = np.count_nonzero(target_boundary & pred_neighbourhood) / max(1, np.count_nonzero(target_boundary))
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def evaluate_pair(prediction_map: np.ndarray, target_map: np.ndarray, iou_threshold=0.5, boundary_tolerance=2) -> dict:
    if prediction_map.shape != target_map.shape:
        raise ValueError(f"Map shapes differ: {prediction_map.shape} != {target_map.shape}")
    predicted, target = _instances(prediction_map), _instances(target_map)
    ious = _iou_matrix(predicted, target)
    matches = []
    if ious.size:
        rows, columns = linear_sum_assignment(-ious)
        matches = [(row, column) for row, column in zip(rows, columns) if ious[row, column] >= iou_threshold]
    dice_values, boundary_values = [], []
    for row, column in matches:
        prediction, reference = predicted[row], target[column]
        intersection = np.count_nonzero(prediction & reference)
        dice_values.append(2 * intersection / max(1, np.count_nonzero(prediction) + np.count_nonzero(reference)))
        boundary_values.append(boundary_f1(prediction, reference, boundary_tolerance))
    return {
        "predicted": len(predicted),
        "target": len(target),
        "matched": len(matches),
        "dice_sum": float(sum(dice_values)),
        "bf1_sum": float(sum(boundary_values)),
    }


def evaluate_directories(prediction_dir: str | Path, target_dir: str | Path, *, iou_threshold=0.5, boundary_tolerance=2) -> dict:
    prediction_dir, target_dir = Path(prediction_dir), Path(target_dir)
    totals = {"predicted": 0, "target": 0, "matched": 0, "dice_sum": 0.0, "bf1_sum": 0.0}
    files = sorted(prediction_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG instance maps found in {prediction_dir}")
    evaluated = 0
    for prediction_path in files:
        target_path = target_dir / prediction_path.name
        if not target_path.exists():
            raise FileNotFoundError(f"Missing reference map: {target_path}")
        prediction = cv2.imread(str(prediction_path), cv2.IMREAD_UNCHANGED)
        target = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
        result = evaluate_pair(prediction, target, iou_threshold, boundary_tolerance)
        for key in totals:
            totals[key] += result[key]
        evaluated += 1
    matched = totals["matched"]
    metrics = {
        "images": evaluated,
        "predicted_instances": totals["predicted"],
        "target_instances": totals["target"],
        "matched_instances": matched,
        "precision": matched / max(1, totals["predicted"]),
        "recall": matched / max(1, totals["target"]),
        "dice": totals["dice_sum"] / max(1, matched),
        "boundary_f1": totals["bf1_sum"] / max(1, matched),
        "iou_threshold": iou_threshold,
        "boundary_tolerance_pixels": boundary_tolerance,
    }
    return metrics


def write_metrics(metrics: dict, output: str | Path) -> None:
    Path(output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

