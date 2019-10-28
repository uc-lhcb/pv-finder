from model.efficiency import pv_locations
import numpy as np


def test_pv_location_discovery():
    targets = np.zeros(4000, dtype=np.float32)
    threshold = 0.5
    integral_threshold = 3.0
    min_width = 3

    targets[3:7] = 1.5  # 4.5 found
    targets[12:15] = 1.5  # 13.0 found
    targets[23:27] = 1.5  # 24.5 found
    targets[40:44] = 0.55  # Not found (too low integral)
    targets[60:50] = 0.45  # Not found (too low threshold)
    targets[80:82] = 5.0  # Not found (too narrow)
    targets[90:95] = 1.0  # 92.0 found

    targets[100:110] = float("nan")

    outputs = pv_locations(targets, threshold, integral_threshold, min_width)

    assert set(outputs) == {4.5, 13.0, 24.5, 92.0}
