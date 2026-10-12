#!/usr/bin/env python3
"""Audit full measured camera rotation jitter, including roll, on physical time."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jvs_mock.camera_stability import summarize, summarize_physics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pose-source', choices=['physics', 'telemetry'], default='physics',
                        help='actual per-step camera body, or diagnostic ROS/FK estimate')
    args = parser.parse_args()
    analyze = summarize_physics if args.pose_source == 'physics' else summarize
    reports = [analyze(folder) for folder in args.records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + '\n')
    for report in reports:
        print(json.dumps(report))


if __name__ == '__main__':
    main()
