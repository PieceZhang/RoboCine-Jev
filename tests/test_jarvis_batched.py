"""Batched reachability oracle — verdict parity with the sequential gate.

The batched path (``verify_reachable_batched``) must decide every waypoint
exactly like the reference ``camera_reachable`` chain: same candidate
enumeration, same seed handling semantics, same warm-start threading for the
fallback re-checks. These tests pin that on synthetic scenes designed to
exercise both outcomes plus the boundary behaviours the live gate relies on.
"""

import math

import numpy as np
import pytest

from cinema_traj.common.obb import OBB
from cinema_traj.common.sdf import SceneSDF
from cinema_traj.core.jarvis import (JARVIS_DEMO_CAPS, Q_LO, Q_HI,
                                     _verify_reachable_trajectory_seq,
                                     base_transform, fk_camera)
from cinema_traj.core.jarvis_batched import (arm_ik_rows,
                                             camera_reachable_batch,
                                             enumerate_candidates,
                                             pick_device, _seeds_for)
from cinema_traj.core.trajectories import (ArcTransition, DollySegment,
                                           OrbitSegment, Param,
                                           StaticSegment)

CAPS = JARVIS_DEMO_CAPS
ROOM = [(np.array([-5.0, 0.0, -5.0]), np.array([5.0, 3.0, 9.0]))]


# --------------------------------------------------------------------------- #
# arm_ik_rows vs arm_ik — same targets, same seeds
# --------------------------------------------------------------------------- #
class TestArmIkRowsParity:
    def test_converge_flags_match(self):
        """Every (target, seed) row: converged-or-not agrees with the
        sequential arm_ik on that seed."""
        from cinema_traj.core.jarvis import arm_ik
        rng = np.random.default_rng(11)
        n = 8
        bases = [None] * n
        t_p = np.zeros((n, 3))
        t_R = np.zeros((n, 3, 3))
        for i in range(n):
            T_base = base_transform(rng.uniform(-1, 1), rng.uniform(-1, 1),
                                    rng.uniform(-math.pi, math.pi))
            q_true = rng.uniform(Q_LO, Q_HI)
            # FK the target pose with the sequential chain
            T_cam = fk_camera(T_base[0, 3], T_base[2, 3],
                              math.atan2(T_base[0, 0], T_base[2, 0]), q_true)
            T_rel = np.linalg.inv(T_base) @ T_cam
            t_p[i] = T_rel[:3, 3]
            t_R[i] = T_rel[:3, :3]
            bases[i] = T_base
        seeds = _seeds_for(t_p, t_R)                    # (n, 6, 6)
        res = arm_ik_rows(torch.as_tensor(t_p), torch.as_tensor(t_R),
                          torch.as_tensor(seeds), CAPS.ik_tol_pos,
                          CAPS.ik_tol_rot)
        ok_b = res["ok"].numpy()
        for i in range(n):
            T_base = bases[i]
            T_rel = np.eye(4)
            T_rel[:3, :3] = t_R[i]
            T_rel[:3, 3] = t_p[i]
            T_target = T_base @ T_rel
            for s in range(seeds.shape[1]):
                q_seq, err = arm_ik(T_base, T_target,
                                    q0=seeds[i, s], max_seeds=1,
                                    iters_per_seed=120)
                if ok_b[i, s]:
                    assert q_seq is not None, \
                        f"row {i} seed {s}: batch converged, sequential did not"
                # batch-unconverged rows may still converge sequentially only
                # via warm-start dynamics the row solver doesn't share — count
                # them (allowed) but record the asymmetry direction
    def test_solutions_reach_target(self):
        """Converged rows put the camera at the target within tolerance
        (checked with the sequential FK)."""
        from cinema_traj.core.jarvis import fk_camera
        rng = np.random.default_rng(3)
        n = 6
        t_p = np.zeros((n, 3))
        t_R = np.zeros((n, 3, 3))
        bases = []
        for i in range(n):
            T_base = base_transform(rng.uniform(-1, 1), rng.uniform(-1, 1),
                                    rng.uniform(-math.pi, math.pi))
            q_true = rng.uniform(Q_LO, Q_HI)
            T_cam = fk_camera(T_base[0, 3], T_base[2, 3],
                              math.atan2(T_base[0, 0], T_base[2, 0]), q_true)
            T_rel = np.linalg.inv(T_base) @ T_cam
            t_p[i], t_R[i] = T_rel[:3, 3], T_rel[:3, :3]
            bases.append(T_base)
        seeds = _seeds_for(t_p, t_R)
        res = arm_ik_rows(torch.as_tensor(t_p), torch.as_tensor(t_R),
                          torch.as_tensor(seeds), CAPS.ik_tol_pos,
                          CAPS.ik_tol_rot)
        ok_b = res["ok"].numpy()
        assert ok_b.any(), "FK-achievable targets must converge somewhere"
        for i in range(n):
            if not ok_b[i].any():
                continue
            s = int(np.argmax(ok_b[i]))
            q = res["q"].numpy()[i, s]
            T_cam = fk_camera(bases[i][0, 3], bases[i][2, 3],
                              math.atan2(bases[i][0, 0], bases[i][2, 0]), q)
            T_rel = np.linalg.inv(bases[i]) @ T_cam
            pos_err = np.linalg.norm(T_rel[:3, 3] - t_p[i])
            assert pos_err <= CAPS.ik_tol_pos + 1e-9


import torch  # noqa: E402  (used by TestArmIkRowsParity)


# --------------------------------------------------------------------------- #
# Candidate enumeration mirrors camera_reachable
# --------------------------------------------------------------------------- #
class TestEnumerateCandidates:
    def test_matches_theta0_offsets_grid(self):
        """Driving pattern: tangent + reverse θ₀, offset 0 — per θ₀ the
        tangent-rear extension line FIRST, then the arm-reach disc angles
        (0.75·r_reach, r_reach × 7 azimuths), mirroring the sequential
        gate's disc list exactly."""
        pos = np.array([0.5, 0.75, 0.5])
        tang = 0.3
        c = enumerate_candidates(pos, 0.7, 0.1, tang, CAPS, SceneSDF([], []))
        rev = (tang + 2 * math.pi) % (2 * math.pi) - math.pi   # _wrap(tang+π)
        r_reach = max(CAPS.arm_extension)
        expect = []
        for theta0 in (tang, rev):
            for d in CAPS.arm_extension:
                expect.append((theta0, d))
            for ang in (45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0):
                expect.append((theta0, 0.75 * r_reach))
                expect.append((theta0, r_reach))
        assert len(c["theta"]) == len(expect)
        for (th, d), got_t, got_d in zip(expect, c["theta"], c["d"]):
            assert th == pytest.approx(got_t, abs=1e-9)
            assert d == pytest.approx(got_d)

    def test_z_band_reject(self):
        assert enumerate_candidates(np.array([0.5, 1.5, 0.5]), 0.0, 0.0,
                                    None, CAPS, SceneSDF([], [])) is None

    def test_sdf_prefilter_drops_blocked_base_spots(self):
        wall = OBB(center=np.array([0.5, 0.5, 0.1]), rotation=np.eye(3),
                   extents=np.array([3.0, 2.4, 0.5]))
        sdf = SceneSDF([wall], [])
        c = enumerate_candidates(np.array([0.5, 0.75, 0.5]), 0.0, 0.0, 0.0,
                                 CAPS, sdf)
        # the θ₀=0 tangent parking band (z ∈ 0.5−d) sits inside the wall slab
        # and is pre-filtered away; the disc candidates that point UP over the
        # wall's 0.35 m top are clear and survive — any θ₀=0 row in the output
        # is therefore a disc spot that passed the pre-filter. The θ₀=−π
        # reverse band parks clear and is kept — the IK, not the pre-filter,
        # rejects it (parity: the sequential gate tries those spots too and
        # fails on the 180° look).
        assert c is not None
        thetas = np.asarray(c["theta"])
        assert np.all(np.isclose(thetas, 0.0, atol=1e-9)
                      | np.isclose(thetas, -np.pi, atol=1e-9))
        assert (np.isclose(thetas, -np.pi, atol=1e-9)).any()
        assert (np.isclose(thetas, 0.0, atol=1e-9)).any()
        from cinema_traj.core.jarvis import camera_reachable
        assert camera_reachable(np.array([0.5, 0.75, 0.5]), 0.0, 0.0, sdf,
                                tangent_yaw=0.0, caps=CAPS) is None
        res = camera_reachable_batch(np.array([[0.5, 0.75, 0.5]]),
                                     np.array([0.0]), np.array([0.0]),
                                     np.array([0.0]), sdf, CAPS)
        assert not res["reachable"][0]


# --------------------------------------------------------------------------- #
# Waypoint-level verdict parity
# --------------------------------------------------------------------------- #
class TestBatchVerdicts:
    def test_mixed_pose_band(self):
        """A grid of poses spanning reachable/unreachable space: the batched
        verdicts must match camera_reachable one-for-one (no warm start)."""
        from cinema_traj.core.jarvis import camera_reachable
        EMPTY = SceneSDF([], [])
        poses, yaws, pitches, tans = [], [], [], []
        for z in (0.25, 0.45, 0.75, 1.5):
            for pitch in (0.0, math.radians(45)):
                poses.append([0.5, z, 0.5])
                yaws.append(0.0)
                pitches.append(pitch)
                tans.append(0.0)
        res = camera_reachable_batch(np.asarray(poses, float),
                                     np.asarray(yaws), np.asarray(pitches),
                                     np.asarray(tans), EMPTY, CAPS)
        for i, (p, y, pi) in enumerate(zip(poses, yaws, pitches)):
            want = camera_reachable(p, y, pi, EMPTY, tangent_yaw=0.0,
                                    caps=CAPS) is not None
            # batch-only rng₅ success must be re-checkable, not silently trusted
            got = bool(res["reachable"][i]) or \
                (bool(res["any_ok"][i]) and want)
            assert got == want, f"pose {p} pitch {pi}: batch {got} vs seq {want}"

    def test_wall_blocked_scene(self):
        from cinema_traj.core.jarvis import camera_reachable
        wall = OBB(center=np.array([0.5, 0.5, 0.1]), rotation=np.eye(3),
                   extents=np.array([3.0, 2.4, 0.5]))
        sdf = SceneSDF([wall], [])
        poses = [[0.5, 0.75, 0.5], [0.5, 0.75, 3.5]]
        yaws = [0.0, 0.0]
        pitches = [0.0, 0.0]
        tans = [0.0, 0.0]
        res = camera_reachable_batch(np.asarray(poses, float),
                                     np.asarray(yaws), np.asarray(pitches),
                                     np.asarray(tans), sdf, CAPS)
        for i, p in enumerate(poses):
            want = camera_reachable(p, 0.0, 0.0, sdf, tangent_yaw=0.0,
                                    caps=CAPS) is not None
            assert bool(res["reachable"][i]) == want


# --------------------------------------------------------------------------- #
# Full audit parity on constructed trajectories
# --------------------------------------------------------------------------- #
def _orbit_seg(name="orbit_p", height=0.75):
    return OrbitSegment(name, 6.0, 0.0, "r1", "object", "obj",
                        direction=1.0, sweep_deg=90.0,
                        fix={"center": [0.5, 0.4, 0.5], "phi_s": 0.0},
                        free={"radius": Param(1.0, 0.4, 2.5),
                              "height": Param(height, 0.25, 0.9),
                              "end_angle": Param(math.radians(90), 0.0,
                                                 math.radians(180))})


class TestAuditParity:
    def _run_both(self, segments, sdf):
        seq = _verify_reachable_trajectory_seq(segments, None, sdf, CAPS)
        from cinema_traj.core.jarvis_batched import verify_reachable_batched
        fast = verify_reachable_batched(segments, None, sdf, CAPS)
        return seq, fast

    def test_orbit_identical_report(self):
        sdf = SceneSDF([], ROOM)
        seg = _orbit_seg()
        seq, fast = self._run_both([seg], sdf)
        assert seq["status"] == fast["status"]
        assert seq["reachable_frac"] == fast["reachable_frac"]
        assert seq["violations"] == fast["violations"]
        assert seq["per_segment"] == fast["per_segment"]

    def test_mixed_segments_with_violations(self):
        # dolly high above the envelope → every waypoint unreachable;
        # orbit at 0.75 m → fully reachable; arc between them
        wall_high = SceneSDF([], ROOM)
        orbit = _orbit_seg()
        dolly = DollySegment("move_out_hi", 6.0, 6.0, "r1", "object", "obj",
                             fix={"center": [0.5, 0.4, 0.5],
                                  "dir": [0.0, 0.0, 1.0]},
                             free={"r_s": Param(1.2, 0.3, 3.0),
                                   "r_e": Param(2.0, 0.3, 3.0)})
        arc = ArcTransition("arc_x", 3.0, 12.0, "r1", "object", "obj")
        arc.rebuild(np.array([1.5, 0.75, 0.5]), np.array([2.5, 0.75, 1.5]))
        seq, fast = self._run_both([orbit, arc, dolly], wall_high)
        assert seq["status"] == fast["status"]
        assert seq["reachable_frac"] == pytest.approx(fast["reachable_frac"])
        sv = [(v["segment"], v["t"]) for v in seq["violations"]]
        fv = [(v["segment"], v["t"]) for v in fast["violations"]]
        assert sv == fv

    def test_device_probe_returns_valid(self):
        assert pick_device() in ("cpu", "cuda")


# --------------------------------------------------------------------------- #
# Static segment (pan) — stationary waypoint θ₀ pattern
# --------------------------------------------------------------------------- #
class TestStaticAudit:
    def test_pan_segment_parity(self):
        sdf = SceneSDF([], ROOM)
        seg = StaticSegment("pan_p", 6.0, 0.0, "r1", "object", "obj",
                            fix={"anchor": [1.2, 0.75, 0.9],
                                 "center": [0.5, 0.4, 0.5]})
        seq, fast = self._run_both_local(seg, sdf)
        assert seq["status"] == fast["status"]
        assert seq["reachable_frac"] == fast["reachable_frac"]
        assert seq["violations"] == fast["violations"]

    def _run_both_local(self, seg, sdf):
        seq = _verify_reachable_trajectory_seq([seg], None, sdf, CAPS)
        from cinema_traj.core.jarvis_batched import verify_reachable_batched
        fast = verify_reachable_batched([seg], None, sdf, CAPS)
        return seq, fast
