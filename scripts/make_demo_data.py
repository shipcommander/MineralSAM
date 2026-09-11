"""Generate a deterministic synthetic YOLO instance-segmentation dataset."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw


def ellipse_polygon(cx: float, cy: float, rx: float, ry: float, angle: float, points: int = 32):
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    polygon = []
    for index in range(points):
        theta = 2.0 * math.pi * index / points
        x0, y0 = rx * math.cos(theta), ry * math.sin(theta)
        polygon.append((cx + x0 * cos_a - y0 * sin_a, cy + x0 * sin_a + y0 * cos_a))
    return polygon


def make_image(path: Path, label_path: Path, seed: int, size: int) -> None:
    rng = random.Random(seed)
    image = Image.new("RGB", (size, size), (28, 34, 38))
    draw = ImageDraw.Draw(image)
    labels = []
    centers = []
    for _ in range(rng.randint(5, 9)):
        for _attempt in range(100):
            rx = rng.uniform(size * 0.045, size * 0.10)
            ry = rng.uniform(size * 0.035, size * 0.085)
            cx = rng.uniform(rx + 4, size - rx - 4)
            cy = rng.uniform(ry + 4, size - ry - 4)
            if all((cx - x) ** 2 + (cy - y) ** 2 > (rx + r + 5) ** 2 for x, y, r in centers):
                break
        centers.append((cx, cy, max(rx, ry)))
        polygon = ellipse_polygon(cx, cy, rx, ry, rng.uniform(0, math.pi))
        colour = tuple(rng.randint(90, 235) for _ in range(3))
        draw.polygon(polygon, fill=colour, outline=(235, 235, 225), width=2)
        normalized = " ".join(f"{value / size:.6f}" for point in polygon for value in point)
        labels.append(f"0 {normalized}")
    image.save(path)
    label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")


def generate(root: Path, train_count: int = 8, val_count: int = 4, size: int = 256) -> None:
    for split, count, offset in (("train", train_count, 0), ("val", val_count, 10_000)):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            stem = f"synthetic_{split}_{index:03d}"
            make_image(image_dir / f"{stem}.png", label_dir / f"{stem}.txt", offset + index, size)
    dataset = (
        f"path: {root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/val\n"
        "names:\n  0: grain\n"
    )
    (root / "dataset.yaml").write_text(dataset, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    generate(args.output, size=args.size)
    print(f"Synthetic dataset written to {args.output.resolve()}")


if __name__ == "__main__":
    main()

