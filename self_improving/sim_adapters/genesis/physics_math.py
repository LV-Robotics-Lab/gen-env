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


def free_fall_step(cfg):
    """Speed picked up in one unsupported step: the floor under every rest criterion.

    A body at rest under mesh contact loses its contact set for single steps, and that
    step reads exactly this value. Any instantaneous-speed limit at or below it is
    unsatisfiable by construction, so the scene gets blamed for a discretisation artefact.
    """
    return abs(cfg["gravity"][2]) * cfg["dt"]


def creep_step(cfg):
    """Distance an unsupported body falls in one step when velocity is zeroed each sample."""
    return 0.5 * abs(cfg["gravity"][2]) * cfg["dt"] ** 2


def stiffness_floor(cfg):
    """Genesis raises constraint_timeconst to twice the step; below that the solve is unstable."""
    return 2 * cfg["dt"]


def effective_speed(state, radius):
    """max(|v|, r*|w|), the sweep-radius weighting Genesis uses for its own rest test.

    One linear tolerance then covers sliding and spinning alike, so a small body's
    rotational jitter no longer needs a second, separately guessed angular limit.
    """
    return float(
        max(
            np.linalg.norm(state["velocity"]),
            radius * np.linalg.norm(state["angular_velocity"]),
        )
    )


def sweep_radius(points):
    """Largest lever arm of a body's own geometry about its origin (Genesis' dof_length)."""
    data = np.asarray(points, dtype=float)
    if data.ndim != 2 or data.shape[1] != 3 or not np.isfinite(data).all() or not len(data):
        raise ValueError("invalid geometry for sweep radius")
    return float(np.linalg.norm(data, axis=1).max())


def longest_run(flags):
    """Longest consecutive True run.

    Genesis calls a body asleep only after its speed stays low for consecutive steps, and
    the same reasoning applies inverted: an isolated sample above a speed limit is the
    contact-dropout artefact, while real motion holds for a run of steps.
    """
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest
