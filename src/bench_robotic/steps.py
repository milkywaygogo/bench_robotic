"""Pure NumPy steps of the ingest pipeline: alignment, resampling, trimming, stats.

These run inside Ray tasks but take and return plain arrays, so they can be
tested without Ray, video files, or a dataset.
"""

from __future__ import annotations

import numpy as np


def time_grid(t_start: float, t_end: float, fps: float) -> np.ndarray:
    """Build evenly spaced timestamps from ``t_start`` up to ``t_end``.

    Parameters
    ----------
    t_start : float
        First timestamp, in seconds.
    t_end : float
        Latest timestamp the grid may reach, in seconds. It is included when
        it falls on the grid.
    fps : float
        Grid rate, in samples per second. Must be positive.

    Returns
    -------
    numpy.ndarray
        Timestamps in seconds, ``1 / fps`` apart.

    Raises
    ------
    ValueError
        ``fps`` is not positive.

    Examples
    --------
    >>> time_grid(1.0, 1.1, 20).round(3).tolist()
    [1.0, 1.05, 1.1]
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    count = int(np.floor((t_end - t_start) * fps + 1e-6)) + 1
    return t_start + np.arange(count) / fps


def interpolate(t_src: np.ndarray, values: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Linearly interpolate each column of ``values`` onto new timestamps.

    Parameters
    ----------
    t_src : numpy.ndarray
        Timestamps of the rows of ``values``, increasing. Shape ``(n,)``.
    values : numpy.ndarray
        Samples to interpolate. Shape ``(n, columns)``.
    t_dst : numpy.ndarray
        Timestamps to interpolate onto. Shape ``(m,)``.

    Returns
    -------
    numpy.ndarray
        Interpolated samples as float32. Shape ``(m, columns)``.

    Examples
    --------
    >>> interpolate(np.array([0.0, 1.0]), np.array([[0.0], [2.0]]), np.array([0.25]))
    array([[0.5]], dtype=float32)
    """
    out = np.empty((len(t_dst), values.shape[1]), dtype=np.float32)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(t_dst, t_src, values[:, col])
    return out


def nearest(t_src: np.ndarray, t_dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the closest ``t_src`` sample for each ``t_dst``.

    Parameters
    ----------
    t_src : numpy.ndarray
        Source timestamps. Must be sorted and hold at least two values.
    t_dst : numpy.ndarray
        Timestamps to match.

    Returns
    -------
    idx : numpy.ndarray
        Index into ``t_src`` of the closest sample, one per ``t_dst``.
    err : numpy.ndarray
        Absolute time difference to that sample, in the units of the inputs.

    Examples
    --------
    >>> idx, err = nearest(np.array([0.0, 0.1, 0.2]), np.array([0.04, 0.16]))
    >>> idx.tolist(), err.round(2).tolist()
    ([0, 2], [0.04, 0.04])
    """
    right = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
    left = right - 1
    pick_left = (t_dst - t_src[left]) <= (t_src[right] - t_dst)
    idx = np.where(pick_left, left, right)
    return idx, np.abs(t_src[idx] - t_dst)


def active_range(
    action: np.ndarray,
    fps: float,
    idle_speed: float,
    pad_s: float,
    smooth_s: float = 0.2,
) -> tuple[int, int]:
    """Find the range of samples where the commanded joints move.

    Teleoperation recordings often start and end with the robot standing
    still. Speed is the largest joint speed, averaged over ``smooth_s`` so
    sensor noise on single steps doesn't count as motion. If nothing moves,
    everything is kept.

    Parameters
    ----------
    action : numpy.ndarray
        Commanded joint positions. Shape ``(n, joints)``.
    fps : float
        Sample rate of ``action``, in samples per second.
    idle_speed : float
        Smoothed speed, in units per second, above which a sample is moving.
    pad_s : float
        Seconds of stillness to keep on each side of the motion.
    smooth_s : float, default 0.2
        Length of the averaging window, in seconds.

    Returns
    -------
    start : int
        Index of the first sample to keep.
    end : int
        Index one past the last sample to keep.

    Examples
    --------
    >>> a = np.array([[0.0]] * 5 + [[float(i)] for i in range(5)] + [[4.0]] * 5)
    >>> active_range(a, fps=10, idle_speed=1.0, pad_s=0.1, smooth_s=0.1)
    (4, 11)
    """
    speed = np.abs(np.diff(action, axis=0)).max(axis=1) * fps
    window = max(int(round(smooth_s * fps)), 1)
    speed = np.convolve(speed, np.ones(window) / window, mode="same")
    moving = np.flatnonzero(speed > idle_speed)
    if len(moving) == 0:
        return 0, len(action)
    pad = int(round(pad_s * fps))
    return int(max(moving[0] - pad, 0)), int(min(moving[-1] + 2 + pad, len(action)))


def partial_stats(x: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-column sums that can be merged across episodes.

    Parameters
    ----------
    x : numpy.ndarray
        Samples. Shape ``(n, columns)``.

    Returns
    -------
    dict of str to numpy.ndarray
        ``count`` (one element), and per-column ``sum``, ``sumsq``, ``min``,
        and ``max``. Pass a list of these to `merge_stats`.

    Examples
    --------
    >>> part = partial_stats(np.array([[1.0, 2.0], [3.0, 6.0]]))
    >>> part["sum"].tolist(), part["max"].tolist()
    ([4.0, 8.0], [3.0, 6.0])
    """
    x = x.astype(np.float64)
    return {
        "count": np.array([len(x)], dtype=np.float64),
        "sum": x.sum(axis=0),
        "sumsq": np.square(x).sum(axis=0),
        "min": x.min(axis=0),
        "max": x.max(axis=0),
    }


def merge_stats(parts: list[dict[str, np.ndarray]]) -> dict[str, list[float]]:
    """Combine `partial_stats` results into normalization statistics.

    Parameters
    ----------
    parts : list of dict
        Results of `partial_stats`, all with the same columns.

    Returns
    -------
    dict
        Per-column ``mean``, ``std``, ``min``, and ``max`` as lists, and the
        total row ``count`` as an int.

    Examples
    --------
    >>> a = partial_stats(np.array([[1.0], [3.0]]))
    >>> b = partial_stats(np.array([[5.0]]))
    >>> merged = merge_stats([a, b])
    >>> merged["mean"], [round(v, 3) for v in merged["std"]], merged["count"]
    ([3.0], [1.633], 3)
    """
    count = sum(float(p["count"][0]) for p in parts)
    total = np.sum([p["sum"] for p in parts], axis=0)
    sumsq = np.sum([p["sumsq"] for p in parts], axis=0)
    mean = total / count
    var = np.maximum(sumsq / count - np.square(mean), 0.0)
    return {
        "mean": mean.tolist(),
        "std": np.sqrt(var).tolist(),
        "min": np.min([p["min"] for p in parts], axis=0).tolist(),
        "max": np.max([p["max"] for p in parts], axis=0).tolist(),
        "count": int(count),
    }
