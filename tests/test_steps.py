from __future__ import annotations

import numpy as np
import pytest

from bench_robotic import source, steps


def test_resample_50_to_30_hz_hits_nearest_frames():
    src = np.arange(400) / 50.0
    grid = steps.time_grid(src[0], src[-1], 30)
    assert len(grid) == 240
    idx, err = steps.nearest(src, grid)
    assert err.max() <= 0.5 / 50 + 1e-9
    assert np.all(np.diff(idx) >= 1)


def test_interpolate_is_exact_for_linear_signals():
    t = np.linspace(0, 1, 11)
    values = np.stack([2 * t, -t + 3], axis=1)
    dst = np.array([0.05, 0.55, 0.95])
    np.testing.assert_allclose(
        steps.interpolate(t, values, dst), np.stack([2 * dst, -dst + 3], 1), rtol=1e-6
    )


def test_active_range_keeps_everything_when_still():
    assert steps.active_range(np.zeros((20, 3)), 10, 0.1, 0.1) == (0, 20)


def test_merge_stats_matches_numpy():
    rng = np.random.default_rng(0)
    parts = [rng.normal(size=(n, 4)) for n in (5, 17, 3)]
    merged = steps.merge_stats([steps.partial_stats(p) for p in parts])
    full = np.concatenate(parts)
    np.testing.assert_allclose(merged["mean"], full.mean(0))
    np.testing.assert_allclose(merged["std"], full.std(0))
    np.testing.assert_allclose(merged["min"], full.min(0))
    assert merged["count"] == 25


def test_time_grid_rejects_non_positive_fps():
    with pytest.raises(ValueError, match="got 0"):
        steps.time_grid(0.0, 1.0, 0)


def test_encode_rejects_unknown_codec(tmp_path):
    out = tmp_path / "videos" / "a.mp4"
    with pytest.raises(ValueError, match="vp9"):
        source.encode([], out, 30, "vp9", 30, "8", 1, 2)
    assert not out.parent.exists()
