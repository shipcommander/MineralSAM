"""Ultralytics integration for per-image AFSS sampling."""

from __future__ import annotations

import os
from copy import copy
from pathlib import Path

import numpy as np

from .afss import AFSSConfig, AFSSScheduler, instance_sufficiency


def normalize_sample_id(path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def dataset_sample_ids(dataset) -> list[str]:
    files = list(getattr(dataset, "im_files", ()) or ())
    if len(files) != len(dataset):
        return [f"dataset-index:{index}" for index in range(len(dataset))]
    return [normalize_sample_id(path) for path in files]


class MutableSubsetDataset:
    """Dataset view whose active indices can change between epochs."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.sample_ids = tuple(dataset_sample_ids(dataset))
        self.index_by_id = {sample_id: index for index, sample_id in enumerate(self.sample_ids)}
        self.active_indices = list(range(len(self.sample_ids)))

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, index):
        return self.dataset[self.active_indices[index]]

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def set_active_ids(self, sample_ids) -> None:
        indices = [self.index_by_id[item] for item in sample_ids if item in self.index_by_id]
        if not indices:
            raise RuntimeError("AFSS selected an empty dataset")
        self.active_indices = indices


class AFSSMetricCollectorMixin:
    """Collect image-level sufficiency during a normal validation pass."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.afss_scores = {}

    def update_metrics(self, predictions, batch):
        super().update_metrics(predictions, batch)
        for sample_index, prediction in enumerate(predictions):
            prepared_batch = self._prepare_batch(sample_index, batch)
            prepared_prediction = self._prepare_pred(prediction)
            processed = self._process_batch(prepared_prediction, prepared_batch)
            box_matches = np.asarray(processed.get("tp", ()), dtype=bool)
            mask_matches = np.asarray(processed.get("tp_m", ()), dtype=bool)
            prediction_count = int(prepared_prediction["cls"].shape[0])
            target_count = int(prepared_batch["cls"].shape[0])
            box_tp = int(box_matches[:, 0].sum()) if box_matches.ndim == 2 and box_matches.shape[1] else 0
            mask_tp = int(mask_matches[:, 0].sum()) if mask_matches.ndim == 2 and mask_matches.shape[1] else 0
            sample_id = normalize_sample_id(prepared_batch["im_file"])
            self.afss_scores[sample_id] = instance_sufficiency(
                box_tp, mask_tp, prediction_count, target_count
            )


def build_metric_validator(trainer):
    if str(getattr(trainer.args, "task", "detect")) == "segment":
        from ultralytics.models.yolo.segment import SegmentationValidator as BaseValidator
    else:
        from ultralytics.models.yolo.detect import DetectionValidator as BaseValidator

    class AFSSMetricValidator(AFSSMetricCollectorMixin, BaseValidator):
        pass

    arguments = copy(trainer.args)
    arguments.plots = False
    arguments.save_json = False
    arguments.save_txt = False
    arguments.verbose = False
    arguments.augment = False
    arguments.split = "train"
    return AFSSMetricValidator(
        trainer._afss_eval_loader,
        save_dir=Path(trainer.save_dir) / "afss_eval",
        args=arguments,
        _callbacks=None,
    )


class AFSSTrainerMixin:
    """Mixin placed before an Ultralytics detection or segmentation trainer."""

    mineralsam_afss_config = AFSSConfig(enabled=False)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        config = AFSSConfig.from_value(self.mineralsam_afss_config)
        if mode != "train" or not config.enabled:
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        from ultralytics.data import build_dataloader
        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        with torch_distributed_zero_first(rank):
            base_dataset = self.build_dataset(dataset_path, mode, batch_size)
            evaluation_dataset = self.build_dataset(dataset_path, "val", batch_size)
        dataset_view = MutableSubsetDataset(base_dataset)
        self._afss_dataset_view = dataset_view
        self._afss_eval_loader = build_dataloader(
            evaluation_dataset,
            batch=min(batch_size, len(evaluation_dataset)),
            workers=0,
            shuffle=False,
            rank=-1,
            drop_last=False,
        )
        self._afss_scheduler = AFSSScheduler(
            dataset_view.sample_ids,
            config,
            state_path=Path(self.save_dir) / "afss_state.json",
        )
        return build_dataloader(
            dataset_view,
            batch=batch_size,
            workers=self.args.workers,
            shuffle=True,
            rank=rank,
            drop_last=self.args.compile,
        )

    def afss_on_epoch_start(self):
        scheduler = getattr(self, "_afss_scheduler", None)
        dataset_view = getattr(self, "_afss_dataset_view", None)
        if scheduler is None or dataset_view is None:
            return None
        selected = scheduler.select(self.epoch, min_count=max(1, int(self.batch_size)))
        dataset_view.set_active_ids(selected)
        self.train_loader.reset()
        summary = scheduler.summary(selected)
        print(
            f"[AFSS] epoch {self.epoch + 1}: {summary['selected']}/{summary['total']} images; "
            f"easy={summary['easy']}, moderate={summary['moderate']}, hard={summary['hard']}"
        )
        return summary

    def afss_on_epoch_end(self):
        scheduler = getattr(self, "_afss_scheduler", None)
        completed_epoch = self.epoch + 1
        total_epochs = max(1, int(getattr(self, "epochs", completed_epoch)))
        if scheduler is None or completed_epoch >= total_epochs or not scheduler.should_update(completed_epoch):
            return None
        validator = build_metric_validator(self)
        validator(trainer=self)
        updated = scheduler.update_scores(validator.afss_scores)
        summary = scheduler.summary()
        print(
            f"[AFSS] refreshed {updated}/{summary['total']} scores; "
            f"easy={summary['easy']}, moderate={summary['moderate']}, hard={summary['hard']}"
        )
        return summary

