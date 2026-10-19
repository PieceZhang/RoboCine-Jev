#!/usr/bin/env python3
"""Independently recompute mixed-frame EE errors from measured world poses."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
args = parser.parse_args()
rows = [json.loads(line) for line in (args.run / "measurements.jsonl").read_text().splitlines()]
summary = json.loads((args.run / "summary.json").read_text())


def rotation(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]])


position_differences, orientation_differences = [], []
for row in rows:
    ee = np.asarray(row["ee_world_pose_xyz_wxyz"])
    link0 = np.asarray(row["link0_world_pose_xyz_wxyz"])
    command = np.asarray(row["ee_command_mixed"])
    assert np.isfinite(np.concatenate([ee, link0, command])).all()
    local_position = rotation(link0[3:]).inv().apply(ee[:3] - link0[:3])
    position_error = np.linalg.norm([local_position[0] - command[0], local_position[1] - command[1], ee[2] - command[2]])
    desired = rotation(link0[3:]) * rotation(command[3:])
    orientation_error = (desired.inv() * rotation(ee[3:])).magnitude()
    position_differences.append(abs(position_error - row["ee_position_error_m"]))
    orientation_differences.append(abs(orientation_error - row["ee_orientation_error_rad"]))
expected = sum(case["expected_control_steps"] for case in summary["cases"])
report = {"measured_rows": len(rows), "expected_rows": expected,
          "position_error_max_recompute_difference_m": float(max(position_differences)),
          "orientation_error_max_recompute_difference_rad": float(max(orientation_differences))}
report["pass"] = bool(len(rows) == expected and max(position_differences) < 1e-4 and max(orientation_differences) < 1e-4)
(args.run / "independent_pose_audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["pass"] else 2)
