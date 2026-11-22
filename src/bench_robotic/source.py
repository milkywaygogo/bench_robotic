"""Read a LeRobot v3.0 dataset: episode index, joint streams, and camera videos.

In v3.0 many episodes share one Parquet file and one MP4 per camera.
``meta/episodes`` gives each episode's row range and its time window in each
video file.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

CODECS = ("av1", "h264")


def load_info(root: Path) -> dict:
    """Load the dataset's ``meta/info.json``.

    Parameters
    ----------
    root : pathlib.Path
        Dataset directory.

    Returns
    -------
    dict
        Parsed metadata: features, fps, robot type, and path templates.
    """
    return json.loads((root / "meta" / "info.json").read_text())


def cameras(info: dict) -> list[str]:
    """List the camera feature names of a dataset.

    Parameters
    ----------
    info : dict
        Metadata from `load_info`.

    Returns
    -------
    list of str
        Names of the features stored as video, in metadata order.

    Examples
    --------
    >>> features = {"cam": {"dtype": "video"}, "action": {"dtype": "float32"}}
    >>> cameras({"features": features})
    ['cam']
    """
    return [k for k, v in info["features"].items() if v["dtype"] == "video"]


def list_episodes(root: Path) -> list[dict]:
    """List every episode with the files and ranges that hold its data.

    Parameters
    ----------
    root : pathlib.Path
        Dataset directory.

    Returns
    -------
    list of dict
        One flat dict per episode, ready for ``ray.data.from_items``. Keys are
        ``episode_index``, ``task``, ``data_path``, ``from_index``,
        ``to_index``, and per camera ``<camera>/path``, ``<camera>/from_ts``,
        and ``<camera>/to_ts``.
    """
    info = load_info(root)
    cams = cameras(info)
    rows = []
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        for ep in pq.read_table(path).to_pylist():
            row = {
                "episode_index": ep["episode_index"],
                "task": ep["tasks"][0],
                "data_path": str(
                    root
                    / info["data_path"].format(
                        chunk_index=ep["data/chunk_index"],
                        file_index=ep["data/file_index"],
                    )
                ),
                "from_index": ep["dataset_from_index"],
                "to_index": ep["dataset_to_index"],
            }
            for cam in cams:
                row[f"{cam}/path"] = str(
                    root
                    / info["video_path"].format(
                        video_key=cam,
                        chunk_index=ep[f"videos/{cam}/chunk_index"],
                        file_index=ep[f"videos/{cam}/file_index"],
                    )
                )
                row[f"{cam}/from_ts"] = ep[f"videos/{cam}/from_timestamp"]
                row[f"{cam}/to_ts"] = ep[f"videos/{cam}/to_timestamp"]
            rows.append(row)
    return rows


def read_joints(episode: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one episode's joint stream.

    Parameters
    ----------
    episode : dict
        One row from `list_episodes`.

    Returns
    -------
    t : numpy.ndarray
        Timestamps in seconds from the episode start. Shape ``(n,)``.
    state : numpy.ndarray
        Measured joint positions. Shape ``(n, joints)``.
    action : numpy.ndarray
        Commanded joint positions. Shape ``(n, joints)``.
    """
    columns = ["index", "timestamp", "observation.state", "action"]
    table = pq.read_table(episode["data_path"], columns=columns)
    index = table["index"]
    mask = pc.and_(
        pc.greater_equal(index, episode["from_index"]),
        pc.less(index, episode["to_index"]),
    )
    table = table.filter(mask).sort_by("index")
    t = table["timestamp"].to_numpy().astype(np.float64)
    state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False))
    action = np.stack(table["action"].to_numpy(zero_copy_only=False))
    return t, state, action


def _open_window(path: str, start: float):
    container = av.open(path)
    stream = container.streams.video[0]
    container.seek(int(start / stream.time_base), stream=stream, backward=True)
    return container, stream


def frame_times(path: str, start: float, end: float) -> np.ndarray:
    """Read the presentation times of the frames in a time window.

    Times come from the packets, so no pixels are decoded.

    Parameters
    ----------
    path : str
        Video file.
    start : float
        Start of the window, in seconds from the start of the file.
    end : float
        End of the window, in seconds. Frames at ``end`` are left out.

    Returns
    -------
    numpy.ndarray
        Sorted frame times in seconds, relative to ``start``.
    """
    container, stream = _open_window(path, start)
    half = 0.5 / float(stream.average_rate)
    times = []
    with container:
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            t = float(packet.pts * stream.time_base)
            if t >= end - half:
                break
            if t >= start - half:
                times.append(t - start)
    return np.sort(np.array(times))


def decode_window(path: str, start: float, end: float) -> list[av.VideoFrame]:
    """Decode the frames in a time window.

    Parameters
    ----------
    path : str
        Video file.
    start : float
        Start of the window, in seconds from the start of the file.
    end : float
        End of the window, in seconds. Frames at ``end`` are left out.

    Returns
    -------
    list of av.VideoFrame
        Decoded frames in presentation order.
    """
    container, stream = _open_window(path, start)
    stream.thread_type = "AUTO"
    half = 0.5 / float(stream.average_rate)
    frames = []
    with container:
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t >= end - half:
                break
            if t >= start - half:
                frames.append(frame)
    return frames


def encode(
    frames: list[av.VideoFrame],
    path: Path,
    fps: int,
    codec: str,
    crf: int,
    preset: str,
    threads: int,
    gop: int,
) -> None:
    """Write frames as one MP4 at a constant frame rate.

    Parameters
    ----------
    frames : list of av.VideoFrame
        Frames to write, in order. Their timestamps are overwritten.
    path : pathlib.Path
        Output file. Parent directories are created.
    fps : int
        Output frame rate.
    codec : str
        ``"av1"`` (SVT-AV1) or ``"h264"`` (x264).
    crf : int
        Constant rate factor. Lower is higher quality and a larger file.
    preset : str
        Encoder speed preset, such as ``"8"`` for AV1 or ``"veryfast"`` for
        H.264.
    threads : int
        Encoder threads.
    gop : int
        Frames between keyframes.

    Raises
    ------
    ValueError
        ``codec`` is not one of `CODECS`.
    """
    if codec not in CODECS:
        raise ValueError(f"codec must be one of {CODECS}, got {codec!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        if codec == "av1":
            stream = out.add_stream("libsvtav1", rate=fps)
            stream.options = {
                "crf": str(crf),
                "preset": preset,
                "svtav1-params": f"lp={threads}",
            }
        else:
            stream = out.add_stream("libx264", rate=fps)
            stream.options = {"crf": str(crf), "preset": preset}
            stream.codec_context.thread_count = threads
        stream.width = frames[0].width
        stream.height = frames[0].height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = gop
        time_base = Fraction(1, fps)
        for i, frame in enumerate(frames):
            frame.pts = i
            frame.time_base = time_base
            out.mux(stream.encode(frame))
        out.mux(stream.encode())


def count_frames(path: Path) -> int:
    """Count the frames in a video without decoding them.

    Parameters
    ----------
    path : pathlib.Path
        Video file.

    Returns
    -------
    int
        Number of packets that carry a presentation time.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return sum(1 for p in container.demux(stream) if p.pts is not None)
