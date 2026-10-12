#!/usr/bin/env python3
"""fp_mux — re-mux first-person PNG frames into a paced mp4.

Fallback for the Isaac-side recorder: the optical camera captures at the
simulation loop rate (~3 fps), not the requested 30 fps, and a signal can
kill jvs_simview_isaac before cv2.VideoWriter.release() leaves fp_cam.mp4
headerless. Frame mtimes are the capture timestamps, so this rebuilds a
true-paced 30 fps mp4 by nearest-neighbour resampling of the mtimes.

Usage:
    python scripts/fp_mux.py <record_dir> [--fps 30] [--out fp_cam.mp4]
                             [--from EPOCH] [--to EPOCH]

--from/--to trim by wall-clock epoch seconds (e.g. the tracker log's
trigger and pass-complete timestamps).
"""

import argparse
import os
from pathlib import Path
import sys

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_dir")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--out", default="fp_cam.mp4")
    parser.add_argument("--from", dest="t_from", type=float, default=None)
    parser.add_argument("--to", dest="t_to", type=float, default=None)
    args = parser.parse_args()

    record_dir = Path(args.record_dir).expanduser().resolve()
    frames = sorted(record_dir.glob("fp_cam_*.png"))
    if not frames:
        print(f"no fp_cam_*.png under {record_dir}", file=sys.stderr)
        return 1
    times = [os.path.getmtime(path) for path in frames]
    t0, t1 = times[0], times[-1]
    t_from = max(t0, args.t_from) if args.t_from is not None else t0
    t_to = min(t1, args.t_to) if args.t_to is not None else t1
    if t_to <= t_from:
        print(f"empty span: {t_from}..{t_to}", file=sys.stderr)
        return 1

    probe = cv2.imread(str(frames[0]))
    height, width = probe.shape[:2]
    video_path = record_dir / args.out
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    written = 0
    index = 0
    target = t_from
    while target <= t_to:
        while index + 1 < len(times) and abs(times[index + 1] - target) <= abs(
            times[index] - target
        ):
            index += 1
        writer.write(cv2.imread(str(frames[index])))
        written += 1
        target += 1.0 / args.fps
    writer.release()
    _transcode_h264(video_path)
    print(
        f"{video_path}: {written} frames @ {args.fps:g}fps "
        f"({written / args.fps:.1f}s real-time pace, {width}x{height})"
    )
    return 0


def _transcode_h264(video_path: Path) -> None:
    """cv2's mp4v (MPEG-4 Part 2) is unplayable in browsers/GitHub preview.

    Re-encode in place to H.264 + yuv420p + faststart when a static ffmpeg
    is available (pip imageio-ffmpeg); silently keep the mp4v otherwise.
    """
    try:
        import imageio_ffmpeg
    except ImportError:
        return
    import subprocess

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = video_path.with_suffix(".h264.tmp.mp4")
    ret = subprocess.run(
        [exe, "-y", "-loglevel", "error", "-i", str(video_path),
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(tmp)],
    ).returncode
    if ret == 0 and tmp.stat().st_size > 0:
        tmp.replace(video_path)


if __name__ == "__main__":
    raise SystemExit(main())
