#!/usr/bin/env python3
"""Start Isaac Python with its bundled ROS 2 ABI isolated from system ROS."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-python", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    return parser


def python_prefix(executable: str) -> Path:
    result = subprocess.run(
        [executable, "-c", "import sys; print(sys.prefix)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"cannot execute Isaac Python {executable!r}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return Path(result.stdout.strip()).resolve()


def bundled_ros_library(prefix: Path) -> Path:
    pattern = "lib/python*/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib"
    candidates = sorted({path.resolve() for path in prefix.glob(pattern) if path.is_dir()})
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one Isaac Sim Humble bridge library under "
            f"{prefix}, found {len(candidates)}"
        )
    return candidates[0]


def child_environment(prefix: Path) -> dict[str, str]:
    environment = os.environ.copy()
    original_pythonpath = environment.pop("PYTHONPATH", "")
    environment["JVS_XACRO_PYTHONPATH"] = original_pythonpath
    ros_library = bundled_ros_library(prefix)
    old_library_path = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = (
        str(ros_library)
        if not old_library_path
        else f"{ros_library}{os.pathsep}{old_library_path}"
    )
    environment.setdefault("ROS_DISTRO", "humble")
    environment.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    return environment


def main() -> int:
    args = build_parser().parse_args()
    executable = str(Path(args.isaac_python).expanduser().resolve())
    runner = str(Path(args.runner).expanduser().resolve())
    if not Path(executable).is_file():
        raise FileNotFoundError(f"Isaac Python does not exist: {executable}")
    if not Path(runner).is_file():
        raise FileNotFoundError(f"Isaac runner does not exist: {runner}")
    runner_args = args.runner_args[1:] if args.runner_args[:1] == ["--"] else args.runner_args
    prefix = python_prefix(executable)
    environment = child_environment(prefix)
    os.execve(executable, [executable, runner, *runner_args], environment)
    return 1  # pragma: no cover - os.execve replaces this process


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"jvs_isaac_bootstrap: {error}", file=sys.stderr)
        raise SystemExit(2)
