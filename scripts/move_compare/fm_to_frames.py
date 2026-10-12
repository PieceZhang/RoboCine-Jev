#!/usr/bin/env python3
"""fm retimed samples -> traj_final-style exec frames (view/render contract).

Samples: {t, x, y, z, yaw_deg, pitch_deg, roll_deg, fov_deg, segment}
Frames:  {t, pos, look, fov_deg, segment}  (densified @10 Hz, look = pos + fwd)
"""
import json
import math
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
s = sorted(d["samples"], key=lambda r: r["t"])
ts = [r["t"] for r in s]
xyz = np.array([[r["x"], r["y"], r["z"]] for r in s], float)
yaw = np.array([r["yaw_deg"] for r in s], float)
pit = np.array([r["pitch_deg"] for r in s], float)
fov = np.array([r.get("fov_deg", 55.0) for r in s], float)
segs = [r.get("segment", "fm") for r in s]

frames = []
t, t1, k = ts[0], ts[-1], 0
while t <= t1 + 1e-9:
    while k + 1 < len(ts) and ts[k + 1] <= t + 1e-9:
        k += 1
    p = [float(np.interp(t, ts, xyz[:, j])) for j in range(3)]
    ya = math.radians(float(np.interp(t, ts, yaw)))
    pi = math.radians(float(np.interp(t, ts, pit)))
    fwd = [math.cos(pi) * math.sin(ya), math.sin(pi), math.cos(pi) * math.cos(ya)]
    frames.append({
        "t": round(t, 3),
        "pos": [round(v, 6) for v in p],
        "look": [round(p[j] + fwd[j], 6) for j in range(3)],
        "fov_deg": round(float(np.interp(t, ts, fov)), 2),
        "segment": segs[k],
    })
    t += 0.1

json.dump({"scene_id": d.get("scene_id", ""),
           "_comment": "fm traj -> exec frames (view/render contract, 10 Hz)",
           "frames": frames}, open(dst, "w"))
print(f"{dst}: {len(frames)} frames, {ts[-1] - ts[0]:.1f}s")
