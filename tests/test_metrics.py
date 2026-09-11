import numpy as np

from mineralsam.metrics import evaluate_pair


def test_identical_instance_maps_are_perfect():
    label_map = np.zeros((32, 32), dtype=np.uint16)
    label_map[3:12, 4:13] = 1
    label_map[18:29, 17:28] = 2

    result = evaluate_pair(label_map, label_map)

    assert result["predicted"] == result["target"] == result["matched"] == 2
    assert result["dice_sum"] == 2.0
    assert result["bf1_sum"] == 2.0

