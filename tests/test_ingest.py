"""End-to-end run of the Ray pipeline on a tiny synthetic LeRobot v3.0 dataset."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import av
import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ray

import bench_robotic
from bench_robotic.ingest import Config, run_ingest
from bench_robotic.sample import run_sample

FPS = 50
CAMS = ["observation.images.cam_a", "observation.images.cam_b"]
LENGTHS = [100, 75]  # 2 s and 1.5 s at 50 Hz; the second is under 1 s once trimmed


def _write_video(path, total, shade):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.codec_context.gop_size = 1
        for i in range(total):
            img = np.full((48, 64, 3), (shade + i) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, FPS)
            out.mux(stream.encode(frame))
        out.mux(stream.encode())


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("src")
    total = sum(LENGTHS)
    features = {c: {"dtype": "video", "shape": [48, 64, 3]} for c in CAMS}
    features["observation.state"] = {"dtype": "float32", "shape": [2]}
    features["action"] = {"dtype": "float32", "shape": [2]}
    info = {
        "robot_type": "toy",
        "fps": FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/"
        "file-{file_index:03d}.mp4",
        "features": features,
    }
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))

    rows, episodes, start = [], [], 0
    for ep, n in enumerate(LENGTHS):
        t = np.arange(n) / FPS
        # Still for 0.4 s, then a ramp, then still for 0.4 s again.
        motion = np.clip((t - 0.4) / (t[-1] - 0.8), 0, 1)
        for i in range(n):
            rows.append(
                {
                    "index": start + i,
                    "timestamp": float(t[i]),
                    "observation.state": [float(motion[i]), 0.0],
                    "action": [float(motion[i]), 1.0],
                }
            )
        ep_row = {
            "episode_index": ep,
            "tasks": ["move"],
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": start,
            "dataset_to_index": start + n,
        }
        for c in CAMS:
            ep_row[f"videos/{c}/chunk_index"] = 0
            ep_row[f"videos/{c}/file_index"] = 0
            ep_row[f"videos/{c}/from_timestamp"] = start / FPS
            ep_row[f"videos/{c}/to_timestamp"] = (start + n) / FPS
        episodes.append(ep_row)
        start += n
    data = root / "data" / "chunk-000" / "file-000.parquet"
    data.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), data)
    pq.write_table(
        pa.Table.from_pylist(episodes),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    for k, c in enumerate(CAMS):
        _write_video(root / "videos" / c / "chunk-000" / "file-000.mp4", total, 40 * k)
    return root


@pytest.fixture(scope="module")
def ingested(dataset, tmp_path_factory):
    package_root = str(Path(bench_robotic.__file__).resolve().parents[1])
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        runtime_env={"env_vars": {"PYTHONPATH": package_root}},
    )
    out = tmp_path_factory.mktemp("out") / "toy"
    cfg = Config(
        src=dataset,
        out=out,
        fps=20,
        codec="h264",
        preset="ultrafast",
        idle_speed=0.1,
        pad_s=0.0,
        encode_cpus=1,
    )
    yield out, run_ingest(cfg)
    ray.shutdown()


def test_steps_videos_and_trimming(ingested):
    out, report = ingested
    assert report["all_videos_match_steps"]
    assert report["flagged"] == {1: "too_short"}
    episodes = lance.dataset(out / "episodes.lance").to_table().to_pylist()
    for ep in episodes:
        # The 0.4 s of stillness at each end is trimmed, give or take smoothing.
        assert 0.25 <= ep["trim_start_s"] <= 0.45
        length = LENGTHS[ep["episode_index"]] / FPS
        assert length - 0.5 <= ep["trim_end_s"] <= length - 0.3
    steps = lance.dataset(out / "steps.lance").to_table()
    assert steps.num_rows == report["output_steps"] == sum(e["steps"] for e in episodes)
    assert steps.schema.field("state").type == pa.list_(pa.float32(), 2)


def test_stats_and_sample(ingested):
    out, _ = ingested
    stats = json.loads((out / "meta" / "stats.json").read_text())
    assert stats["action"]["min"][1] == stats["action"]["max"][1] == 1.0
    sample = run_sample(out, count=3, seed=1)
    assert sample["state_shape"] == [3, 2]
    for cam in CAMS:
        assert sample["images"][cam] == [3, 48, 64, 3]
