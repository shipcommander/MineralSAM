"""Model-independent Adaptive Fine-grained Sample Scheduling (AFSS)."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional


@dataclass(frozen=True)
class AFSSConfig:
    enabled: bool = True
    warmup_epochs: int = 5
    easy_threshold: float = 0.85
    hard_threshold: float = 0.55
    easy_fraction: float = 0.02
    moderate_fraction: float = 0.40
    easy_review_interval: int = 10
    moderate_review_interval: int = 3
    update_interval: int = 5
    max_forced_easy_share: float = 0.50
    min_samples_per_epoch: int = 1
    seed: int = 20260318

    @classmethod
    def from_value(cls, value=None, **overrides) -> "AFSSConfig":
        if isinstance(value, cls):
            data = asdict(value)
        elif isinstance(value, Mapping):
            data = {key: item for key, item in value.items() if key in cls.__dataclass_fields__}
        else:
            data = {}
        data.update({key: item for key, item in overrides.items() if item is not None})
        return cls(**data).validated()

    def validated(self) -> "AFSSConfig":
        hard = max(0.0, min(1.0, float(self.hard_threshold)))
        easy = max(hard, min(1.0, float(self.easy_threshold)))
        return AFSSConfig(
            enabled=bool(self.enabled),
            warmup_epochs=max(0, int(self.warmup_epochs)),
            easy_threshold=easy,
            hard_threshold=hard,
            easy_fraction=max(0.0, min(1.0, float(self.easy_fraction))),
            moderate_fraction=max(0.0, min(1.0, float(self.moderate_fraction))),
            easy_review_interval=max(1, int(self.easy_review_interval)),
            moderate_review_interval=max(1, int(self.moderate_review_interval)),
            update_interval=max(1, int(self.update_interval)),
            max_forced_easy_share=max(0.0, min(1.0, float(self.max_forced_easy_share))),
            min_samples_per_epoch=max(1, int(self.min_samples_per_epoch)),
            seed=int(self.seed),
        )


@dataclass
class SampleState:
    score: float = 0.0
    last_used_epoch: int = -1
    update_count: int = 0


class AFSSScheduler:
    """Select samples using periodically refreshed learning-sufficiency scores."""

    state_version = 1

    def __init__(
        self,
        sample_ids: Iterable[str],
        config: AFSSConfig | Mapping | None = None,
        *,
        state_path: Optional[str | Path] = None,
    ):
        self.sample_ids = tuple(dict.fromkeys(str(item) for item in sample_ids))
        if not self.sample_ids:
            raise ValueError("AFSS requires at least one sample ID")
        self.config = AFSSConfig.from_value(config)
        self.state_path = Path(state_path).resolve() if state_path else None
        self.states = {sample_id: SampleState() for sample_id in self.sample_ids}
        self.last_epoch = -1
        self.last_selected = self.sample_ids
        self._load()

    def should_update(self, completed_epoch: int) -> bool:
        completed_epoch = int(completed_epoch)
        if not self.config.enabled or completed_epoch < self.config.warmup_epochs:
            return False
        return (completed_epoch - self.config.warmup_epochs) % self.config.update_interval == 0

    def update_scores(self, scores: Mapping[str, float]) -> int:
        updated = 0
        for sample_id, raw_score in scores.items():
            state = self.states.get(str(sample_id))
            if state is None:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            state.score = max(0.0, min(1.0, score))
            state.update_count += 1
            updated += 1
        self.save()
        return updated

    def difficulty(self, sample_id: str) -> str:
        score = self.states[str(sample_id)].score
        if score > self.config.easy_threshold:
            return "easy"
        if score >= self.config.hard_threshold:
            return "moderate"
        return "hard"

    def select(self, epoch: int, *, min_count: Optional[int] = None) -> list[str]:
        epoch = int(epoch)
        required = max(1, int(min_count or self.config.min_samples_per_epoch))
        if not self.config.enabled or epoch < self.config.warmup_epochs:
            selected = list(self.sample_ids)
        else:
            groups = {"easy": [], "moderate": [], "hard": []}
            for sample_id in self.sample_ids:
                groups[self.difficulty(sample_id)].append(sample_id)
            generator = random.Random(self.config.seed + epoch)
            selected = list(groups["hard"])
            selected.extend(self._sample_group(groups["moderate"], self.config.moderate_fraction, self.config.moderate_review_interval, epoch, generator))
            selected.extend(self._sample_easy(groups["easy"], epoch, generator))
            if len(set(selected)) < min(required, len(self.sample_ids)):
                selected_set = set(selected)
                remaining = [item for item in self.sample_ids if item not in selected_set]
                remaining.sort(key=lambda item: (self.states[item].score, self.states[item].last_used_epoch, item))
                selected.extend(remaining[: required - len(selected_set)])
        selected_set = set(selected)
        ordered = [item for item in self.sample_ids if item in selected_set]
        for sample_id in ordered:
            self.states[sample_id].last_used_epoch = epoch
        self.last_epoch = epoch
        self.last_selected = tuple(ordered)
        self.save()
        return ordered

    def _sample_group(self, items, fraction, interval, epoch, generator) -> list[str]:
        if not items or fraction <= 0:
            return []
        target = max(1, math.ceil(len(items) * fraction))
        forced = [item for item in items if epoch - self.states[item].last_used_epoch >= interval]
        forced.sort(key=lambda item: (self.states[item].last_used_epoch, item))
        remainder = [item for item in items if item not in set(forced)]
        generator.shuffle(remainder)
        return forced + remainder[: max(0, target - len(forced))]

    def _sample_easy(self, items, epoch, generator) -> list[str]:
        if not items or self.config.easy_fraction <= 0:
            return []
        target = max(1, math.ceil(len(items) * self.config.easy_fraction))
        due = [item for item in items if epoch - self.states[item].last_used_epoch >= self.config.easy_review_interval]
        due.sort(key=lambda item: (self.states[item].last_used_epoch, item))
        forced_budget = 0
        if due and self.config.max_forced_easy_share > 0:
            forced_budget = min(target, max(1, math.floor(target * self.config.max_forced_easy_share)))
        forced = due[:forced_budget]
        remainder = [item for item in items if item not in set(forced)]
        generator.shuffle(remainder)
        return forced + remainder[: max(0, target - len(forced))]

    def summary(self, selected_ids: Optional[Iterable[str]] = None) -> dict:
        counts = {"easy": 0, "moderate": 0, "hard": 0}
        for sample_id in self.sample_ids:
            counts[self.difficulty(sample_id)] += 1
        selected = tuple(selected_ids) if selected_ids is not None else self.last_selected
        return {**counts, "selected": len(selected), "total": len(self.sample_ids)}

    def save(self) -> None:
        if self.state_path is None:
            return
        payload = {
            "version": self.state_version,
            "config": asdict(self.config),
            "sample_ids": list(self.sample_ids),
            "last_epoch": self.last_epoch,
            "states": {sample_id: asdict(state) for sample_id, state in self.states.items()},
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(f"{self.state_path}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, self.state_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _load(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(payload.get("version", 0)) != self.state_version:
                return
            stored = payload.get("states", {})
            for sample_id, state in self.states.items():
                record = stored.get(sample_id)
                if not isinstance(record, dict):
                    continue
                state.score = max(0.0, min(1.0, float(record.get("score", 0.0))))
                state.last_used_epoch = int(record.get("last_used_epoch", -1))
                state.update_count = max(0, int(record.get("update_count", 0)))
            self.last_epoch = int(payload.get("last_epoch", -1))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return


def instance_sufficiency(
    box_true_positives: int,
    mask_true_positives: int,
    prediction_count: int,
    target_count: int,
) -> float:
    """Compute `min(P_box, R_box, P_mask, R_mask)` for one image."""
    prediction_count = max(0, int(prediction_count))
    target_count = max(0, int(target_count))
    if target_count == 0:
        return 1.0 if prediction_count == 0 else 0.0
    box_tp = max(0, min(int(box_true_positives), prediction_count, target_count))
    mask_tp = max(0, min(int(mask_true_positives), prediction_count, target_count))
    return float(
        min(
            box_tp / max(1, prediction_count),
            box_tp / target_count,
            mask_tp / max(1, prediction_count),
            mask_tp / target_count,
        )
    )

