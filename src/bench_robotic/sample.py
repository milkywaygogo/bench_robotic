"""Read training samples back: what a data loader does with the ingest output.

A sample is one step: its state and action rows from `steps.lance`, plus
frame `step_index` of each camera's MP4 for that episode. The short GOP set at
ingest means each frame needs only a seek and a couple of decoded frames.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import av
import lance
import numpy as np


def read_frame(path: Path, index: int, fps: int) -> np.ndarray:
    """Decode one frame of a constant-rate MP4.

    Parameters
    ----------
    path : pathlib.Path
        Video file.
    index : int
        Frame number, from 0.
    fps : int
        Frame rate of the video.

    Returns
    -------
    numpy.ndarray
        RGB image as uint8. Shape ``(height, width, 3)``.

    Raises
    ------
    IndexError
        The video has no frame ``index``.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        target = index / fps
        container.seek(int(target / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time >= target - 0.5 / fps:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(f"frame {index} not found in {path}")


def save_mosaic(images: dict[str, np.ndarray], path: Path) -> None:
    """Save the first sample's camera views as one half-size 2-column image.

    Parameters
    ----------
    images : dict of str to numpy.ndarray
        RGB images by camera name. Shape ``(samples, height, width, 3)``.
    path : pathlib.Path
        Image file to write. The extension picks the format.
    """
    tiles = [x[0, ::2, ::2] for x in images.values()]
    tiles += [np.zeros_like(tiles[0])] * (len(tiles) % 2)
    rows = [np.concatenate(tiles[i : i + 2], axis=1) for i in range(0, len(tiles), 2)]
    av.VideoFrame.from_ndarray(np.concatenate(rows), format="rgb24").to_image().save(
        path
    )


def run_sample(dest: Path, count: int, seed: int, image: Path | None = None) -> dict:
    """Read random steps from an ingested dataset, as a training loader would.

    Parameters
    ----------
    dest : pathlib.Path
        Output directory of `bench_robotic.ingest.run_ingest`.
    count : int
        Number of steps to read.
    seed : int
        Seed for choosing the steps.
    image : pathlib.Path, optional
        When given, save the first sample's camera views to this file.

    Returns
    -------
    dict
        A JSON-ready report: the chosen ``rows``, their ``episode_step``
        pairs, array shapes, ``take_ms`` for the Lance read, and
        ``decode_ms_per_frame``.
    """
    info = json.loads((dest / "meta" / "info.json").read_text())
    stats = json.loads((dest / "meta" / "stats.json").read_text())
    steps = lance.dataset(dest / "steps.lance")
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(steps.count_rows(), size=count, replace=False))

    began = time.perf_counter()
    table = steps.take(rows.tolist())
    take_s = time.perf_counter() - began

    mean = np.array(stats["observation.state"]["mean"], dtype=np.float32)
    std = np.array(stats["observation.state"]["std"], dtype=np.float32)
    state = np.stack(table["state"].to_numpy(zero_copy_only=False))
    state_norm = (state - mean) / np.maximum(std, 1e-6)

    began = time.perf_counter()
    images = {}
    for cam, spec in info["cameras"].items():
        frames = [
            read_frame(dest / spec["path"].format(episode_index=ep), step, info["fps"])
            for ep, step in zip(
                table["episode_index"].to_pylist(), table["step_index"].to_pylist()
            )
        ]
        images[cam] = np.stack(frames)
    decode_s = time.perf_counter() - began
    if image:
        save_mosaic(images, image)

    return {
        "rows": rows.tolist(),
        "episode_step": list(
            zip(table["episode_index"].to_pylist(), table["step_index"].to_pylist())
        ),
        "state_shape": list(state_norm.shape),
        "state_norm_mean_abs": float(np.abs(state_norm).mean()),
        "images": {cam: list(x.shape) for cam, x in images.items()},
        "take_ms": take_s * 1e3,
        "decode_ms_per_frame": decode_s * 1e3 / (count * len(images)),
    }
