#!/usr/bin/env python3
"""fix_dwell_arcs — interpolate the base stream across DWELL runs in
jarvis_traj.json.

allow_unreachable holds an unreachable transition sample's config at the
previous one (a "dwell"). The frozen BASE line in that run zeroes the
tracker's feedforward and pins its position feedback to the old parking
spot — the relocation never happens and the camera error ramps to metres
(R26/R28 arcs). This tool splices a smooth base interpolation across each
dwell run (positions linear, yaw wrapped-linear, arm q held), restoring a
drivable feedforward. The camera stream (traj_samples_retimed.json) is
untouched — only the whole-body config stream is repaired.

Usage:
    python scripts/fix_dwell_arcs.py <jarvis_traj.json> [--dry]
"""

import argparse
import json
import math


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    doc = json.load(open(args.path))
    cfgs = doc["configs"]
    runs = []                      # [start, end] inclusive index of frozen base
    i = 0
    while i < len(cfgs) - 1:
        j = i + 1
        while j < len(cfgs) and \
                all(abs(cfgs[j]["base"][k] - cfgs[i]["base"][k]) < 1e-9
                    for k in range(3)):
            j += 1
        if j - i >= 8:             # ≥8 frozen samples = a dwell run
            # split the run at segment-label boundaries and keep only the
            # arc-labeled sub-runs: a STATIONARY shooting segment (pan at
            # an anchor) also has a constant base — that is the shot's
            # semantics, not a frozen feedforward. Smoothing across the
            # kid_combo pan's 777 parked configs moved the ff line's end
            # 0.9 m off the parking spot (beyond the 0.6 m arm span) and
            # froze the camera error at ~91 cm for the whole back half.
            k = i
            while k < j:
                lab = str(cfgs[k].get("segment") or "")
                m = k
                while m + 1 < j and str(cfgs[m + 1].get("segment") or "") == lab:
                    m += 1
                if lab.startswith("arc") and m - k >= 4:
                    runs.append((k, m))
                k = m + 1
        i = j
    print(f"{len(runs)} dwell run(s): {[(a, b) for a, b in runs]}")

    n = 0
    for a, b in runs:
        # interpolate from the sample BEFORE the run to the FIRST sample
        # after it when it exists (the run may reach the stream's end)
        i0 = max(a - 1, 0)
        i1 = min(b + 1, len(cfgs) - 1)
        if i1 <= i0:
            continue
        b0, b1 = cfgs[i0]["base"], cfgs[i1]["base"]
        dyaw = wrap(b1[2] - b0[2])
        span = i1 - i0
        # DIFFERENTIAL-BASE yaw profile: a naive linear yaw makes the base
        # translate east while its heading rotates 180° — nonholonomic
        # nonsense; once the heading passes ~90° off the velocity the
        # tracker's forward channel reverses and the robot stalls mid-run
        # (measured: frozen at 1/3 of the migration, pan 2.2 m off).
        # Drive FACING THE TRAVEL DIRECTION, then turn in place to the
        # parking heading over the last 20% of the dwell.
        dxz, dz = b1[0] - b0[0], b1[1] - b0[1]
        travel_yaw = math.atan2(dxz, dz) if math.hypot(dxz, dz) > 1e-3 else b0[2]
        turn_f = 0.8
        for k in range(i0 + 1, i1):
            f = (k - i0) / span
            c = cfgs[k]
            if f <= turn_f:
                yaw_k = travel_yaw if abs(wrap(travel_yaw - b0[2])) < 2.7 else b0[2]
            else:
                yaw_from = travel_yaw if abs(wrap(travel_yaw - b0[2])) < 2.7 else b0[2]
                yaw_k = wrap(yaw_from + (f - turn_f) / (1.0 - turn_f)
                             * wrap(b1[2] - yaw_from))
            c["base"] = [b0[0] + f * dxz,
                         b0[1] + f * dz,
                         yaw_k]
            n += 1
    print(f"interpolated {n} dwell configs")

    if not args.dry and n:
        doc["dwell_arcs_fixed"] = True
        json.dump(doc, open(args.path, "w"), indent=1)
        print(f"written → {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
