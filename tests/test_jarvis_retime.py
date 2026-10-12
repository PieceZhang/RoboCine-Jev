"""Jarvis trackability tests — the retime loop with Jacobian-projected caps,
the ik_stream configuration pipeline, and the jarvis export end-to-end
(including the platform switch in retime_stage)."""

import json
import math

import numpy as np
import pytest

from cinema_traj.common.sdf import SceneSDF
from cinema_traj.core.jarvis import (JARVIS_DEMO_CAPS, Q_HI, Q_LO,
                                     check_trackable, trackability_factor)
from cinema_traj.core.retime import retime_trajectory
from cinema_traj.stages.jarvis_export import ik_stream, jarvis_retime_all
from vlm_intention.common.atomic_io import load_json

CAPS = JARVIS_DEMO_CAPS
EMPTY = SceneSDF([], [])


def _line_samples(speed=2.0, length=4.0, z=0.75, n=11):
    """Straight +Z path at height z, level look — drone-fast, jarvis-slow."""
    out = []
    for k in range(n):
        u = k / (n - 1)
        out.append({"t": length / speed * u, "x": 0.5, "y": z, "z": 0.2 + length * u,
                    "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0,
                    "fov_deg": 55.0, "segment": "test_line"})
    return out


# --------------------------------------------------------------------------- #
# Retime under JarvisCapabilities (shared loop, Jacobian caps)
# --------------------------------------------------------------------------- #
class TestJarvisRetime:
    def test_fast_line_stretched_to_trackable_speed(self):
        samples = _line_samples(speed=2.0)
        res = retime_trajectory(samples, CAPS, keep_duration=False)
        assert res.scale > 1.5                          # 2 m/s ≫ 0.85 cap
        speeds = [math.dist((a["x"], a["z"]), (b["x"], b["z"])) / max(b["t"] - a["t"], 1e-9)
                  for a, b in zip(res.samples, res.samples[1:])]
        assert max(speeds) <= CAPS.max_base_linear_vel + 0.1
        # shape preserved: all samples on the original line x=0.5, y=0.75
        for s in res.samples:
            assert s["x"] == pytest.approx(0.5, abs=1e-9)
            assert s["y"] == pytest.approx(0.75, abs=1e-9)
            assert 0.2 - 1e-6 <= s["z"] <= 4.2 + 1e-6

    def test_stationary_pan_needs_no_stretch(self):
        pan = [{"t": 6.0 * u, "x": 0.5, "y": 0.7, "z": 0.5,
                "yaw_deg": -22.5 + 45.0 * u, "pitch_deg": 0.0, "roll_deg": 0.0,
                "fov_deg": 55.0, "segment": "pan"} for u in np.linspace(0.0, 1.0, 13)]
        res = retime_trajectory(pan, CAPS, keep_duration=True)
        rates = [abs(b["yaw_deg"] - a["yaw_deg"]) / max(b["t"] - a["t"], 1e-9)
                 for a, b in zip(res.samples, res.samples[1:])]
        assert max(rates) <= math.degrees(CAPS.max_yaw_rate) + 1e-6
        assert res.scale == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# ik_stream + exact verification
# --------------------------------------------------------------------------- #
class TestIkStream:
    def test_stream_is_continuous_and_in_limits(self):
        samples = retime_trajectory(_line_samples(speed=2.0), CAPS,
                                    keep_duration=False).samples
        configs, failures, _holds = ik_stream(samples, EMPTY, CAPS)
        assert not failures
        for c in configs:
            q = np.asarray(c["q"])
            assert np.all(q >= Q_LO - 1e-9) and np.all(q <= Q_HI + 1e-9)
        # the base glides monotonically behind the camera (same branch kept)
        zs = [c["base"][1] for c in configs]
        assert all(b >= a - 1e-6 for a, b in zip(zs, zs[1:]))

    def test_trackability_factor_flags_fast_stream(self):
        samples = retime_trajectory(_line_samples(speed=2.0), CAPS,
                                    keep_duration=False).samples
        configs, _failures, _holds = ik_stream(samples, EMPTY, CAPS)
        # the retime loop already stretched to the reference-J cap — the exact
        # factor is ≈1 (small residue allowed); a deliberately fast stream flags
        assert trackability_factor(configs, CAPS) < 1.3
        fast = [{"t": c["t"] * 0.5, "base": c["base"], "q": c["q"]} for c in configs]
        assert trackability_factor(fast, CAPS) > 1.8


# --------------------------------------------------------------------------- #
# jarvis_retime_all end-to-end (empty scene, synthetic traj_final)
# --------------------------------------------------------------------------- #
def _traj_final():
    frames = [{"t": s["t"], "pos": [s["x"], s["y"], s["z"]],
               "look": [s["x"], s["y"], s["z"] + 1.0],   # look +Z: level, yaw 0
               "fov_deg": 55.0, "segment": s["segment"]}
              for s in _line_samples(speed=2.0)]
    return {"scene_id": "jarvis_test", "metrics": {}, "frames": frames}


class TestJarvisRetimeAll:
    def test_end_to_end_outputs_are_trackable(self, tmp_path):
        res = jarvis_retime_all(_traj_final(), tmp_path, CAPS, EMPTY)
        assert res["samples"] >= 10 and res["configs"] == res["samples"]

        jdoc = load_json(res["jarvis_json"])
        assert jdoc["trackable"] is True
        configs = jdoc["configs"]
        assert configs[0]["segment"] == "test_line"
        for c in configs:
            for v in c["q"]:
                assert Q_LO[np.where(np.asarray(c["q"]) == v)[0][0]] - 1e-9 <= v <= \
                    Q_HI[np.where(np.asarray(c["q"]) == v)[0][0]] + 1e-9

        # the camera stream file is schema-compatible with the drone path
        cam = load_json(tmp_path / "exports" / "traj_samples_retimed.json")
        assert cam["frame"]["up"] == "+Y"
        assert cam["retime"]["platform"] == "jarvis"
        assert len(cam["samples"]) == res["samples"]

        # re-verify the written stream exactly
        samples = cam["samples"]
        cfgs = [{"t": c["t"], "base": tuple(c["base"]), "q": np.asarray(c["q"])}
                for c in configs]
        assert check_trackable(samples, cfgs, CAPS) == []

    def test_feedforward_csv_zero_terminal(self, tmp_path):
        res = jarvis_retime_all(_traj_final(), tmp_path, CAPS, EMPTY)
        lines = (tmp_path / "exports" / "jarvis_setpoints.csv").read_text().splitlines()
        # header + one row per config + one parked terminal row
        assert len(lines) == res["configs"] + 2
        last = lines[-1].split(",")
        assert float(last[4]) == 0.0 and float(last[5]) == 0.0  # parked terminal


# --------------------------------------------------------------------------- #
# 分段重定时（杀 ×162 全局膨胀）+ holds 分级
# --------------------------------------------------------------------------- #
class TestPerSegmentRetime:
    def _two_segment_samples(self):
        """A: 水平直线 0.4 m/s（不顶 cap）；B: 竖直上升 0.4 m/s（远超臂载
        0.135 cap → 该段必须大幅拉伸）。全局重定时会把 A 一起拖慢。"""
        A_seg = [{"t": 0.1 * k, "x": 0.5, "y": 0.7, "z": 1.0 + 0.04 * k,
                  "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0,
                  "fov_deg": 55.0, "segment": "move_a"} for k in range(6)]
        B_seg = [{"t": 0.5 + 0.1 * k, "x": 0.5, "y": 0.7 + 0.04 * k, "z": 1.2,
                  "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0,
                  "fov_deg": 55.0, "segment": "arc_b"} for k in range(6)]
        return A_seg + B_seg

    def test_slow_segment_does_not_inflate_neighbors(self):
        from cinema_traj.stages.jarvis_export import _retime_per_segment
        out = _retime_per_segment(self._two_segment_samples(), CAPS, 0.05)
        # monotonic time
        ts = [s["t"] for s in out]
        assert all(b > a for a, b in zip(ts, ts[1:]))
        # segment A keeps its own speed (~0.4 m/s — a global retime would
        # dilute it by B's ~3× stretch)
        a = [s for s in out if s["segment"] == "move_a"]
        speeds = [abs(b["z"] - a_["z"]) / max(b["t"] - a_["t"], 1e-9)
                  for a_, b in zip(a, a[1:])]
        assert max(speeds) > 0.3, f"A got diluted: {max(speeds):.3f}"
        assert max(speeds) <= CAPS.max_base_linear_vel + 0.1
        # segment B stretched well beyond its raw 0.5 s span
        b = [s for s in out if s["segment"] == "arc_b"]
        assert b[-1]["t"] - b[0]["t"] > 1.0

    def test_joined_stream_is_trackable(self):
        from cinema_traj.stages.jarvis_export import _retime_per_segment
        out = _retime_per_segment(self._two_segment_samples(), CAPS, 0.05)
        configs, failures, _holds = ik_stream(out, EMPTY, CAPS)
        assert not failures
        assert check_trackable(out, configs, CAPS) == []


class TestHoldsTiering:
    def _samples(self):
        ok1 = {"t": 0.0, "x": 0.5, "y": 0.75, "z": 1.0, "yaw_deg": 0.0,
               "pitch_deg": 0.0, "roll_deg": 0.0, "fov_deg": 55.0,
               "segment": "move_a"}
        bad_arc = dict(ok1, t=0.1, y=1.5, segment="arc_b")     # unreachable
        bad_shot = dict(ok1, t=0.2, y=1.5, segment="move_a")   # unreachable
        ok2 = dict(ok1, t=0.3)
        return [ok1, bad_arc, bad_shot, ok2]

    def test_arc_holds_but_shooting_fails_even_when_allowed(self):
        configs, failures, holds = ik_stream(self._samples(), EMPTY, CAPS,
                                             allow_unreachable=True)
        assert holds == 1                                   # the arc waypoint
        assert len(failures) == 1
        assert failures[0]["segment"] == "move_a"           # shooting → failure
        assert configs[1] is not None and configs[2] is None

    def test_strict_mode_holds_nothing(self):
        configs, failures, holds = ik_stream(self._samples(), EMPTY, CAPS,
                                             allow_unreachable=False)
        assert holds == 0 and len(failures) == 2
        assert all(c is None for c in (configs[1], configs[2]))


# --------------------------------------------------------------------------- #
# Platform switch — retime_stage routes to the jarvis chain
# --------------------------------------------------------------------------- #
class TestPlatformSwitch:
    def test_retime_stage_routes_to_jarvis(self, tmp_path):
        # room deep enough for the 4 m path AND the trailing base (it parks up
        # to ~0.6 m behind the camera — the base, not the camera, bounds it)
        scene = {"scene_id": "empty_room",
                 "rooms": [{"id": "r1", "label": "room",
                            "bounds_min": [-3.0, 0.0, -3.0],
                            "bounds_max": [3.0, 2.8, 6.0]}]}
        scene_path = tmp_path / "empty_room.json"
        scene_path.write_text(json.dumps(scene), encoding="utf-8")
        traj_path = tmp_path / "traj_final.json"
        traj_path.write_text(json.dumps(_traj_final()), encoding="utf-8")
        cfg = {"cinema_traj": {"platform": "jarvis", "scene": str(scene_path),
                               "outputs_dir": str(tmp_path / "out"),
                               "jarvis": {}, "retime": {}}}
        from cinema_traj.stages.retime_stage import run
        res = run(cfg, traj_path=traj_path, out_dir=tmp_path, force=True)
        assert (tmp_path / "exports" / "jarvis_traj.json").exists()
        assert res["configs"] == res["samples"]
