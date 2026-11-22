"""Command line for the robot data pipeline examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_REPO = "lerobot/aloha_static_cups_open"
DEFAULT_REVISION = "v3.0"  # a tag, so reruns fetch the same files
DEFAULT_ROOT = Path("/data/hf")


def _parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="bench_robotic", description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Download a LeRobot dataset from the Hub")
    dl.add_argument("--repo", default=DEFAULT_REPO)
    dl.add_argument("--revision", default=DEFAULT_REVISION)
    dl.add_argument("--root", type=Path, default=DEFAULT_ROOT)

    ing = sub.add_parser("ingest", help="Run the Ray ingest pipeline")
    ing.add_argument("--src", type=Path, default=DEFAULT_ROOT / DEFAULT_REPO)
    ing.add_argument(
        "--dest", type=Path, default=Path("/data/robotic/aloha_static_cups_open")
    )
    ing.add_argument("--fps", type=int, default=30)
    ing.add_argument("--codec", choices=["av1", "h264"], default="av1")
    ing.add_argument("--crf", type=int, default=30)
    ing.add_argument(
        "--preset", default=None, help="Encoder preset. Defaults: av1 8, h264 veryfast"
    )
    ing.add_argument("--gop", type=int, default=2)
    ing.add_argument("--encode-cpus", type=int, default=2)
    ing.add_argument("--idle-speed", type=float, default=0.2)
    ing.add_argument("--out", type=Path, help="Write the JSON report to this path")

    smp = sub.add_parser(
        "sample", help="Read random training samples back from an ingested dataset"
    )
    smp.add_argument(
        "--dest", type=Path, default=Path("/data/robotic/aloha_static_cups_open")
    )
    smp.add_argument("--count", type=int, default=8)
    smp.add_argument("--seed", type=int, default=0)
    smp.add_argument("--image", type=Path, help="Save the first sample's cameras")
    return root


def main(argv: list[str] | None = None) -> int:
    """Run one ``bench_robotic`` command and print its result.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    args = _parser().parse_args(argv)
    if args.command == "download":
        from huggingface_hub import snapshot_download

        dest = args.root / args.repo
        snapshot_download(
            args.repo, repo_type="dataset", revision=args.revision, local_dir=dest
        )
        print(dest)
        return 0

    if args.command == "sample":
        from bench_robotic.sample import run_sample

        report = run_sample(args.dest, args.count, args.seed, args.image)
        print(json.dumps(report, indent=2))
        return 0

    from bench_robotic.ingest import Config, run_ingest

    cfg = Config(
        src=args.src,
        out=args.dest,
        fps=args.fps,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset or ("8" if args.codec == "av1" else "veryfast"),
        gop=args.gop,
        encode_cpus=args.encode_cpus,
        idle_speed=args.idle_speed,
    )
    report = run_ingest(cfg)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0
