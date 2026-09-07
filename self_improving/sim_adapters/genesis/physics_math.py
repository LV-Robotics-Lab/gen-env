"""Simulator-independent rigid pose calculations shared by both physics entrances."""

import itertools
import math

import numpy as np


def rotation(quat):
    q = np.asarray(quat, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-5:
        raise ValueError("invalid unit quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def corners(bounds, pose):
    b = np.asarray(bounds, dtype=float)
    if b.shape != (2, 3) or not np.isfinite(b).all() or not (b[1] > b[0]).all():
        raise ValueError("invalid local geometry bounds")
    local = np.array(list(itertools.product(*zip(b[0], b[1]))))
    return local @ rotation(pose["orientation_wxyz"]).T + pose["position"]


def angle(a, b):
    rotation(a)
    rotation(b)
    dot = abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(2 * math.acos(np.clip(dot, 0, 1)))
