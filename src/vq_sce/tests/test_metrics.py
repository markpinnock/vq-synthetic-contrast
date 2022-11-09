import numpy as np
import pytest
import sys
import tensorflow as tf

from syntheticcontrast_v02.utils.losses import FocalMetric


def test_metric():
    metric = FocalMetric()
    a = np.zeros((2, 4, 4), dtype=np.float32)
    b = np.ones((2, 4, 4), dtype=np.float32)
    b[0, 1, 1] = 2
    b[1, 1, 1] = 4
    m = np.zeros((2, 4, 4), dtype=np.float32)
    m[:, 1, 1] = 1

    gt_focal = 3
    gt_global = 1
    gt = np.stack([gt_global, gt_focal])
    
    metric.update_state(a, b, m)
    assert np.all(metric.result().numpy() == gt)
    metric.reset_states()
    assert np.all(metric.result().numpy() == np.array([0.0, 0.0]))


def test_metric_loop():
    metric = FocalMetric()
    a = np.zeros((2, 4, 4), dtype=np.float32)
    b = np.ones((2, 4, 4), dtype=np.float32)
    b[0, 1, 1] = 2
    b[1, 1, 1] = 4
    m = np.zeros((2, 4, 4), dtype=np.float32)
    m[:, 1, 1] = 1

    gt_focal = 3
    gt_global = 1
    gt = np.stack([gt_global, gt_focal])
    
    for _ in range(5):
        metric.update_state(a, b, m)

    assert np.isclose(metric.result().numpy(), gt).all()
    metric.reset_states()
    assert np.isclose(metric.result().numpy(), np.array([0.0, 0.0])).all(), metric.result()
