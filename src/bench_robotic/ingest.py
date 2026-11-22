"""Ray Data ingest pipeline: LeRobot episodes in, training-ready dataset out.

The stages::

    meta/episodes ──► align ──► (materialized, one row per episode)
                                 │
                 ┌───────────────┼─────────────────────┐
                 ▼               ▼                     ▼
        encode (one task   steps table          episode table,
        per episode and    (Lance fragments)    stats, info
        camera, AV1/H.264)                      (driver)

1. align: read the joint stream, trim idle time at both ends, resample state
   and action onto an ``fps`` grid, and pick the nearest camera frame for each
   grid step. Also flags quality problems and computes partial statistics.
2. encode: decode each camera's window, keep the chosen frames, and encode
   one MP4 per episode and camera. Frame k of the MP4 belongs to step k.
3. steps: one row per step (state, action, timestamp). Each Ray task writes
   Lance fragments; the driver commits them as one dataset version.
4. driver: merge statistics and write the episode table and metadata.

Output layout::

    OUT/steps.lance                     one row per step
    OUT/episodes.lance                  one row per episode, with quality flags
    OUT/videos/<camera>/episode_000000.mp4
    OUT/meta/info.json                  robot, cameras, fps, tasks, settings
    OUT/meta/stats.json                 normalization statistics
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import ray

from bench_robotic import source, steps

PIXEL_SAMPLE_EVERY = 10  # frames between samples for image statistics


@dataclass
class Config:
    """Settings for one ingest run.

    Parameters
    ----------
    src : pathlib.Path
        LeRobot v3.0 dataset to read.
    out : pathlib.Path
        Output directory. `run_ingest` deletes it first if it exists.
    fps : int, default 30
        Output rate, in steps per second.
    idle_speed : float, default 0.2
        Smoothed commanded joint speed, in rad/s, above which the robot
        counts as moving. Used to trim idle time.
    pad_s : float, default 0.2
        Seconds of stillness kept on each side of the motion.
    codec : str, default "av1"
        Video codec, ``"av1"`` or ``"h264"``.
    crf : int, default 30
        Encoder constant rate factor.
    preset : str, default "8"
        Encoder speed preset.
    gop : int, default 2
        Frames between keyframes. A short GOP lets a training loader seek to
        any frame cheaply.
    encode_cpus : int, default 2
        CPUs Ray reserves for each encode task, and the encoder's threads.
    min_seconds : float, default 1.0
        Episodes shorter than this after trimming are flagged ``too_short``.
    max_joint_step : float, default 0.3
        Joint change between consecutive steps, in rad, above which an
        episode is flagged ``joint_jump``.
    """

    src: Path
    out: Path
    fps: int = 30
    idle_speed: float = 0.2
    pad_s: float = 0.2
    codec: str = "av1"
    crf: int = 30
    preset: str = "8"
    gop: int = 2
    encode_cpus: int = 2
    min_seconds: float = 1.0
    max_joint_step: float = 0.3


def align_episode(ep: dict, cfg: Config, cams: list[str]) -> dict:
    """Trim, resample, and match camera frames for one episode.

    Parameters
    ----------
    ep : dict
        One row from `bench_robotic.source.list_episodes`.
    cfg : Config
        Run settings.
    cams : list of str
        Camera feature names.

    Returns
    -------
    dict
        One row for the aligned dataset: the trim window, resampled
        ``timestamp``, ``state``, and ``action`` arrays, the chosen source
        frame indices as ``<camera>/frames``, ``max_align_error_ms``,
        ``max_joint_step``, comma-separated ``flags``, and partial statistics
        under ``stats/``.
    """
    t, state, action = source.read_joints(ep)
    src_fps = 1.0 / np.median(np.diff(t))
    start, end = steps.active_range(action, src_fps, cfg.idle_speed, cfg.pad_s)
    grid = steps.time_grid(t[start], t[end - 1], cfg.fps)
    state_r = steps.interpolate(t, state, grid)
    action_r = steps.interpolate(t, action, grid)

    out = {
        "episode_index": ep["episode_index"],
        "task": ep["task"],
        "source_steps": len(t),
        "trim_start_s": float(t[start]),
        "trim_end_s": float(t[end - 1]),
        "timestamp": (grid - grid[0]).astype(np.float32),
        "state": state_r,
        "action": action_r,
    }
    worst_ms = 0.0
    for cam in cams:
        pts = source.frame_times(
            ep[f"{cam}/path"], ep[f"{cam}/from_ts"], ep[f"{cam}/to_ts"]
        )
        idx, err = steps.nearest(pts, grid)
        out[f"{cam}/frames"] = idx.astype(np.int32)
        worst_ms = max(worst_ms, float(err.max()) * 1e3)
    out["max_align_error_ms"] = worst_ms

    jump = float(np.abs(np.diff(state_r, axis=0)).max()) if len(grid) > 1 else 0.0
    out["max_joint_step"] = jump
    flags = []
    if len(grid) / cfg.fps < cfg.min_seconds:
        flags.append("too_short")
    if jump > cfg.max_joint_step:
        flags.append("joint_jump")
    if worst_ms > 500.0 / cfg.fps:
        flags.append("camera_gap")
    if not (np.isfinite(state_r).all() and np.isfinite(action_r).all()):
        flags.append("non_finite")
    out["flags"] = ",".join(flags)
    for name, values in (("state", state_r), ("action", action_r)):
        for key, value in steps.partial_stats(values).items():
            out[f"stats/{name}/{key}"] = value
    return out


def camera_jobs(ep: dict, src_eps: dict[int, dict], cams: list[str]) -> list[dict]:
    """Split one aligned episode into one encode job per camera.

    Parameters
    ----------
    ep : dict
        One row from `align_episode`.
    src_eps : dict of int to dict
        Rows from `bench_robotic.source.list_episodes`, by episode index.
    cams : list of str
        Camera feature names.

    Returns
    -------
    list of dict
        One job per camera, with the source video ``path``, its ``from_ts``
        and ``to_ts`` window, and the ``frames`` to keep.
    """
    raw = src_eps[ep["episode_index"]]
    return [
        {
            "episode_index": ep["episode_index"],
            "camera": cam,
            "path": raw[f"{cam}/path"],
            "from_ts": raw[f"{cam}/from_ts"],
            "to_ts": raw[f"{cam}/to_ts"],
            "frames": ep[f"{cam}/frames"],
        }
        for cam in cams
    ]


def encode_camera(job: dict, cfg: Config) -> dict:
    """Decode one camera window, keep the aligned frames, and encode an MP4.

    Parameters
    ----------
    job : dict
        One job from `camera_jobs`.
    cfg : Config
        Run settings.

    Returns
    -------
    dict
        The ``video`` path relative to ``cfg.out``, its ``frames`` and
        ``bytes``, the task time ``encode_s``, and partial pixel statistics
        under ``stats/``.
    """
    began = time.perf_counter()
    decoded = source.decode_window(job["path"], job["from_ts"], job["to_ts"])
    chosen = [decoded[i] for i in job["frames"]]
    pixels = (
        np.stack(
            [f.to_ndarray(format="rgb24") for f in chosen[::PIXEL_SAMPLE_EVERY]]
        ).reshape(-1, 3)
        / 255.0
    )
    rel = f"videos/{job['camera']}/episode_{job['episode_index']:06d}.mp4"
    dest = cfg.out / rel
    source.encode(
        chosen, dest, cfg.fps, cfg.codec, cfg.crf, cfg.preset, cfg.encode_cpus, cfg.gop
    )
    result = {
        "episode_index": job["episode_index"],
        "camera": job["camera"],
        "video": rel,
        "frames": len(chosen),
        "bytes": dest.stat().st_size,
        "encode_s": time.perf_counter() - began,
    }
    for key, value in steps.partial_stats(pixels).items():
        result[f"stats/{key}"] = value
    return result


def step_schema(state_dim: int, action_dim: int) -> pa.Schema:
    """Build the Arrow schema of the steps table.

    Parameters
    ----------
    state_dim : int
        Length of the state vector.
    action_dim : int
        Length of the action vector.

    Returns
    -------
    pyarrow.Schema
        Columns ``episode_index``, ``step_index``, ``timestamp``, ``state``,
        and ``action``.

    Examples
    --------
    >>> step_schema(14, 14).names
    ['episode_index', 'step_index', 'timestamp', 'state', 'action']
    """
    return pa.schema(
        [
            ("episode_index", pa.int64()),
            ("step_index", pa.int32()),
            ("timestamp", pa.float32()),
            ("state", pa.list_(pa.float32(), state_dim)),
            ("action", pa.list_(pa.float32(), action_dim)),
        ]
    )


def to_steps(batch: dict[str, np.ndarray]) -> pa.Table:
    """Explode a batch of aligned episodes into one row per step.

    Lance treats ``.`` in a top-level name as a struct path, so the columns
    are ``state`` and ``action`` rather than LeRobot's ``observation.state``.

    Parameters
    ----------
    batch : dict of str to numpy.ndarray
        A batch of rows from `align_episode`, one array element per episode.

    Returns
    -------
    pyarrow.Table
        One row per step, with the schema from `step_schema`.
    """
    episode, step, ts, state, action = [], [], [], [], []
    for i, ep_index in enumerate(batch["episode_index"]):
        n = len(batch["timestamp"][i])
        episode.append(np.full(n, ep_index, dtype=np.int64))
        step.append(np.arange(n, dtype=np.int32))
        ts.append(batch["timestamp"][i])
        state.append(np.asarray(batch["state"][i], dtype=np.float32))
        action.append(np.asarray(batch["action"][i], dtype=np.float32))
    state_np, action_np = np.concatenate(state), np.concatenate(action)

    def vectors(x: np.ndarray) -> pa.FixedSizeListArray:
        return pa.FixedSizeListArray.from_arrays(pa.array(x.ravel()), x.shape[1])

    return pa.table(
        [
            np.concatenate(episode),
            np.concatenate(step),
            np.concatenate(ts),
            vectors(state_np),
            vectors(action_np),
        ],
        schema=step_schema(state_np.shape[1], action_np.shape[1]),
    )


def write_step_fragments(batch: dict[str, np.ndarray], dest: str) -> dict:
    """Write one batch of episodes as uncommitted Lance fragments.

    Ray's built-in ``write_lance`` does the same, but Ray 2.60 passes an
    argument that pylance 14 no longer accepts.

    Parameters
    ----------
    batch : dict of str to numpy.ndarray
        A batch of rows from `align_episode`.
    dest : str
        URI of the Lance dataset the fragments belong to.

    Returns
    -------
    dict
        ``fragment``: an array of JSON strings, one per written fragment, for
        the driver to commit.
    """
    table = to_steps(batch)
    fragments = lance.fragment.write_fragments(table, dest, schema=table.schema)
    return {"fragment": np.array([json.dumps(f.to_json()) for f in fragments])}


def _stats_of(rows: list[dict], prefix: str) -> list[dict]:
    keys = ("count", "sum", "sumsq", "min", "max")
    return [{k: np.asarray(r[f"{prefix}{k}"]) for k in keys} for r in rows]


def run_ingest(cfg: Config) -> dict:
    """Run the whole pipeline.

    Starts Ray locally if it is not running, and deletes ``cfg.out`` first if
    it exists.

    Parameters
    ----------
    cfg : Config
        Run settings.

    Returns
    -------
    dict
        A JSON-ready report: episode, step, and frame counts, flagged
        episodes, the largest alignment error, video sizes, and ``timings``
        per stage in seconds.
    """
    info = source.load_info(cfg.src)
    cams = source.cameras(info)
    episodes = source.list_episodes(cfg.src)
    src_eps = {e["episode_index"]: e for e in episodes}
    if cfg.out.exists():
        shutil.rmtree(cfg.out)
    cfg.out.mkdir(parents=True)

    # Workers import this package by name, so they need its source directory.
    package_root = str(Path(__file__).resolve().parents[1])
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={"env_vars": {"PYTHONPATH": package_root}},
    )
    timings = {}
    began = time.perf_counter()

    aligned = (
        ray.data.from_items(episodes, override_num_blocks=len(episodes))
        .map(align_episode, fn_kwargs={"cfg": cfg, "cams": cams})
        .materialize()
    )
    timings["align_s"] = time.perf_counter() - began

    t = time.perf_counter()
    videos = (
        aligned.flat_map(camera_jobs, fn_kwargs={"src_eps": src_eps, "cams": cams})
        .repartition(len(episodes) * len(cams))
        .map(encode_camera, fn_kwargs={"cfg": cfg}, num_cpus=cfg.encode_cpus)
        .take_all()
    )
    timings["encode_s"] = time.perf_counter() - t

    t = time.perf_counter()
    steps_uri = str(cfg.out / "steps.lance")
    written = aligned.map_batches(
        write_step_fragments,
        fn_kwargs={"dest": steps_uri},
        batch_format="numpy",
        batch_size=8,
    ).take_all()
    lance.LanceDataset.commit(
        steps_uri,
        lance.LanceOperation.Overwrite(
            step_schema(
                info["features"]["observation.state"]["shape"][0],
                info["features"]["action"]["shape"][0],
            ),
            [lance.FragmentMetadata.from_json(w["fragment"]) for w in written],
        ),
    )
    timings["steps_s"] = time.perf_counter() - t

    t = time.perf_counter()
    arrays = ("state", "action", "timestamp")
    ep_rows = aligned.drop_columns(
        [c for c in aligned.columns() if "/" in c or c in arrays]
    ).take_all()
    all_eps = sorted(aligned.take_all(), key=lambda r: r["episode_index"])
    by_ep: dict[int, dict] = {}
    for v in videos:
        by_ep.setdefault(v["episode_index"], {})[v["camera"]] = v
    table_rows = []
    for row in sorted(ep_rows, key=lambda r: r["episode_index"]):
        e = row["episode_index"]
        row["steps"] = by_ep[e][cams[0]]["frames"]
        row["video_frames_match"] = all(
            by_ep[e][c]["frames"] == row["steps"]
            and source.count_frames(cfg.out / by_ep[e][c]["video"]) == row["steps"]
            for c in cams
        )
        for c in cams:
            row[f"video_{c.rsplit('.', 1)[-1]}"] = by_ep[e][c]["video"]
        table_rows.append(row)
    lance.write_dataset(pa.Table.from_pylist(table_rows), cfg.out / "episodes.lance")

    stats = {
        "observation.state": steps.merge_stats(_stats_of(all_eps, "stats/state/")),
        "action": steps.merge_stats(_stats_of(all_eps, "stats/action/")),
    }
    for c in cams:
        stats[c] = steps.merge_stats(
            _stats_of([v for v in videos if v["camera"] == c], "stats/")
        )
        stats[c]["note"] = f"RGB in [0, 1], every {PIXEL_SAMPLE_EVERY}th frame"
    meta = cfg.out / "meta"
    meta.mkdir()
    (meta / "stats.json").write_text(json.dumps(stats, indent=2))

    total_steps = sum(r["steps"] for r in table_rows)
    out_info = {
        "robot_type": info["robot_type"],
        "fps": cfg.fps,
        "source_fps": info["fps"],
        "total_episodes": len(table_rows),
        "total_steps": total_steps,
        "tasks": sorted({r["task"] for r in table_rows}),
        "cameras": {
            c: {
                "shape": info["features"][c]["shape"],
                "codec": cfg.codec,
                "gop": cfg.gop,
                "path": f"videos/{c}/episode_{{episode_index:06d}}.mp4",
            }
            for c in cams
        },
        "features": {k: info["features"][k] for k in ("observation.state", "action")},
        "source": str(cfg.src),
        "pipeline": {k: str(v) for k, v in asdict(cfg).items()},
    }
    (meta / "info.json").write_text(json.dumps(out_info, indent=2))
    timings["finalize_s"] = time.perf_counter() - t
    timings["total_s"] = time.perf_counter() - began

    source_steps = sum(r["source_steps"] for r in table_rows)
    video_bytes = sum(v["bytes"] for v in videos)
    src_video_bytes = sum(p.stat().st_size for p in (cfg.src / "videos").rglob("*.mp4"))
    return {
        "dataset": str(cfg.src),
        "out": str(cfg.out),
        "cpus": int(ray.cluster_resources().get("CPU", 0)),
        "episodes": len(table_rows),
        "cameras": len(cams),
        "source_steps": source_steps,
        "output_steps": total_steps,
        "flagged": {r["episode_index"]: r["flags"] for r in table_rows if r["flags"]},
        "all_videos_match_steps": all(r["video_frames_match"] for r in table_rows),
        "max_align_error_ms": max(r["max_align_error_ms"] for r in table_rows),
        "trimmed_s": sum(
            r["source_steps"] / info["fps"] - (r["trim_end_s"] - r["trim_start_s"])
            for r in table_rows
        ),
        "video_mb": video_bytes / 1e6,
        "source_video_mb": src_video_bytes / 1e6,
        "frames_encoded": sum(v["frames"] for v in videos),
        "encode_task_s_mean": float(np.mean([v["encode_s"] for v in videos])),
        "timings": timings,
        "pid": os.getpid(),
    }
