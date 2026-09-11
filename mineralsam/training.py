"""Configuration-driven MineralPose training and domain adaptation."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from .adapters import (
    adapter_parameter_ids,
    freeze_for_adapters,
    install_conv_adapters,
    parameter_summary,
    set_batchnorm_eval,
    set_segment_head_trainable,
)
from .afss import AFSSConfig
from .ultralytics_ext import AFSSTrainerMixin


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for key in ("weights", "data"):
        value = config.get(key)
        if value and not Path(value).is_absolute():
            candidate = (config_path.parent.parent / value).resolve()
            if candidate.exists() or key == "data":
                config[key] = str(candidate)
    project = config.get("project")
    if project and not Path(project).is_absolute():
        config["project"] = str((config_path.parent.parent / project).resolve())
    return config


def build_trainer_class(task: str, adapter: dict, afss: dict):
    if task == "segment":
        from ultralytics.models.yolo.segment import SegmentationTrainer as BaseTrainer
    elif task == "detect":
        from ultralytics.models.yolo.detect import DetectionTrainer as BaseTrainer
    else:
        raise ValueError("MineralSAM training supports only 'segment' and 'detect' tasks")

    adapter_settings = dict(adapter)
    afss_config = AFSSConfig.from_value(afss)

    class MineralSAMTrainer(AFSSTrainerMixin, BaseTrainer):
        mineralsam_afss_config = afss_config

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            if adapter_settings.get("enabled", False):
                model.mineralsam_adapter_result = install_conv_adapters(
                    model,
                    target_layers=adapter_settings.get("target_layers", "auto"),
                    bottleneck_ratio=adapter_settings.get("bottleneck_ratio", "dynamic"),
                    residual_scale=float(adapter_settings.get("residual_scale", 0.25)),
                    output_init_std=float(adapter_settings.get("output_init_std", 1e-3)),
                )
            return model

    return MineralSAMTrainer


def configure_adapter_training(trainer, adapter: dict) -> dict:
    """Freeze the host and replace the optimizer with explicit parameter groups."""
    result = getattr(trainer.model, "mineralsam_adapter_result", {})
    if not result.get("n_adapters"):
        raise RuntimeError("No eligible C3k2 layers were found for Adapter insertion")
    freeze_for_adapters(trainer.model)
    adapter_ids = adapter_parameter_ids(trainer.model)
    segment_ids = set()
    if adapter.get("train_segment_head", False):
        segment_ids = set_segment_head_trainable(trainer.model)
    if adapter.get("freeze_batchnorm", True):
        set_batchnorm_eval(trainer.model)

    adapter_parameters, segment_parameters = [], []
    for parameter in trainer.model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in adapter_ids:
            adapter_parameters.append(parameter)
        elif id(parameter) in segment_ids:
            segment_parameters.append(parameter)

    learning_rate = float(trainer.args.lr0)
    momentum = float(trainer.args.momentum)
    groups = []
    if adapter_parameters:
        groups.append(
            {
                "params": adapter_parameters,
                "weight_decay": float(adapter.get("weight_decay", 1e-4)),
            }
        )
    if segment_parameters:
        groups.append(
            {
                "params": segment_parameters,
                "lr": learning_rate * float(adapter.get("segment_head_lr_ratio", 0.03)),
                "weight_decay": 0.0,
            }
        )
    optimizer_name = str(trainer.args.optimizer or "AdamW").lower()
    if optimizer_name == "sgd":
        trainer.optimizer = torch.optim.SGD(
            groups, lr=learning_rate, momentum=momentum, nesterov=True
        )
    elif optimizer_name == "adam":
        trainer.optimizer = torch.optim.Adam(groups, lr=learning_rate, betas=(momentum, 0.999))
    else:
        trainer.optimizer = torch.optim.AdamW(groups, lr=learning_rate, betas=(momentum, 0.999))
    trainer._setup_scheduler()
    trainer.scheduler.last_epoch = trainer.start_epoch - 1
    summary = parameter_summary(trainer.model)
    print(
        f"[Adapter] inserted={result['n_adapters']}, "
        f"trainable={summary['trainable']:,}/{summary['total']:,} "
        f"({summary['ratio'] * 100:.2f}%)"
    )
    return summary


def train_from_config(config: dict):
    from ultralytics import YOLO

    task = str(config.get("task", "segment"))
    mode = str(config.get("mode", "adapter")).lower()
    weights = config["weights"]
    data = config["data"]
    adapter = dict(config.get("adapter") or {})
    afss = dict(config.get("afss") or {})
    train_arguments = dict(config.get("train") or {})
    if mode != "adapter":
        adapter["enabled"] = False

    model = YOLO(weights)
    trainer_class = build_trainer_class(task, adapter, afss)
    if adapter.get("enabled", False):
        model.add_callback(
            "on_pretrain_routine_end",
            lambda trainer: configure_adapter_training(trainer, adapter),
        )
        if adapter.get("freeze_batchnorm", True):
            model.add_callback("on_train_epoch_start", lambda trainer: set_batchnorm_eval(trainer.model))
    model.add_callback(
        "on_train_epoch_start",
        lambda trainer: trainer.afss_on_epoch_start()
        if hasattr(trainer, "afss_on_epoch_start")
        else None,
    )
    model.add_callback(
        "on_train_epoch_end",
        lambda trainer: trainer.afss_on_epoch_end()
        if hasattr(trainer, "afss_on_epoch_end")
        else None,
    )
    return model.train(
        data=data,
        project=config.get("project", "runs"),
        name=config.get("name", mode),
        trainer=trainer_class,
        **train_arguments,
    )
