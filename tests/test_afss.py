from mineralsam.afss import AFSSConfig, AFSSScheduler, instance_sufficiency


def test_paper_difficulty_thresholds_and_sampling():
    identifiers = [f"sample-{index}" for index in range(100)]
    scheduler = AFSSScheduler(
        identifiers,
        AFSSConfig(warmup_epochs=0, easy_fraction=0.02, moderate_fraction=0.40),
    )
    scores = {identifier: 0.20 for identifier in identifiers[:20]}
    scores.update({identifier: 0.70 for identifier in identifiers[20:70]})
    scores.update({identifier: 0.95 for identifier in identifiers[70:]})
    scheduler.update_scores(scores)

    selected = scheduler.select(1)
    summary = scheduler.summary(selected)

    assert summary == {"easy": 30, "moderate": 50, "hard": 20, "selected": 41, "total": 100}
    assert set(identifiers[:20]).issubset(selected)


def test_warmup_uses_all_samples():
    scheduler = AFSSScheduler(["a", "b", "c"], AFSSConfig(warmup_epochs=5))
    scheduler.update_scores({"a": 0.99, "b": 0.99, "c": 0.99})
    assert scheduler.select(4) == ["a", "b", "c"]


def test_segmentation_sufficiency_uses_weakest_measure():
    score = instance_sufficiency(
        box_true_positives=8,
        mask_true_positives=5,
        prediction_count=10,
        target_count=8,
    )
    assert score == 0.5

