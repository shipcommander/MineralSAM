"""Command-line entry points for training, prediction, and evaluation."""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mineralsam")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train or adapt MineralPose")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--weights")
    train_parser.add_argument("--data")

    predict_parser = commands.add_parser("predict", help="Create instance-ID maps")
    predict_parser.add_argument("--weights", required=True)
    predict_parser.add_argument("--source", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--confidence", type=float, default=0.25)
    predict_parser.add_argument("--image-size", type=int, default=512)
    predict_parser.add_argument("--device", default="")
    predict_parser.add_argument("--max-detections", type=int, default=3000)
    predict_parser.add_argument("--sam-checkpoint")
    predict_parser.add_argument("--sam-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")

    evaluate_parser = commands.add_parser("evaluate", help="Evaluate instance-ID maps")
    evaluate_parser.add_argument("--pred", required=True)
    evaluate_parser.add_argument("--target", required=True)
    evaluate_parser.add_argument("--output", default="metrics.json")
    evaluate_parser.add_argument("--iou-threshold", type=float, default=0.5)
    evaluate_parser.add_argument("--boundary-tolerance", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        from .training import load_config, train_from_config

        config = load_config(args.config)
        if args.weights:
            config["weights"] = args.weights
        if args.data:
            config["data"] = args.data
        train_from_config(config)
        return
    if args.command == "predict":
        from .inference import predict

        records = predict(
            args.weights,
            args.source,
            args.output,
            confidence=args.confidence,
            image_size=args.image_size,
            device=args.device,
            max_detections=args.max_detections,
            sam_checkpoint=args.sam_checkpoint,
            sam_type=args.sam_type,
        )
        print(f"Wrote {len(records)} prediction records to {args.output}")
        return
    if args.command == "evaluate":
        from .metrics import evaluate_directories, write_metrics

        metrics = evaluate_directories(
            args.pred,
            args.target,
            iou_threshold=args.iou_threshold,
            boundary_tolerance=args.boundary_tolerance,
        )
        write_metrics(metrics, args.output)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

