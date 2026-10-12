"""Batched reachability oracle — the audit fast path for the Jarvis IK gate.

Why this module exists (measured, kid_combo cProfile): the plan stage spends
~95% of its wall clock inside ``verify_reachable_trajectory → camera_reachable
→ arm_ik``, a per-waypoint Python loop of per-seed damped-least-squares
Newton iterations on 3×3 numpy objects (~750k Jacobian evaluations per run).
The audit only consumes the VERDICT (reachable / not) plus a warm-start joint
vector — never the ordered configuration stream — so the whole search
factorizes into one batched tensor program: every (waypoint × base-candidate ×
IK-seed) row advances its own Newton/active-set state simultaneously.

Exact-parity contract (verdicts identical to the sequential gate):

  * candidate enumeration mirrors ``camera_reachable`` /
    ``_camera_reachable_theta0`` for the audit's calling pattern (driving:
    tangent and reverse-tangent θ₀, offset 0 only — |π| > 2.4 is skipped
    exactly as there — crossed with the d grid, SDF-clearance and room-bounds
    pre-filters; target poses in each candidate base frame are built with the
    SAME numpy helper functions, so the per-target rng digests match
    bit-for-bit);
  * the batched seed set is [mid-range, rng₁..rng₅] — the sequential list when
    no warm start is given.  A batch success via seed index 0..4 (mid|rng₁..₄)
    therefore implies the warm-started sequential gate also succeeds (its set
    is [q0, mid, rng₁..rng₄], a superset);
  * every waypoint the batch leaves unreachable is re-decided by the ORIGINAL
    per-waypoint ``camera_reachable`` with prev-q warm-start threading — the
    sequential algorithm verbatim.  Final verdicts are the sequential gate's.

The IK's floating-point trajectory differs from numpy at ~1e-12 relative
(different op order/BLAS), so a verdict can in principle flip for a target
sitting that close to a convergence boundary; the parity test pins observed
behaviour on synthetic scenes.

Device selection: CUDA is probed by EXECUTING a tiny float64 op (an
``is_available()``-only probe happily reports True on sm_120 GPUs under
cu126-only torch, then every kernel launch fails).  ``ROBOCINE_IK_DEVICE=
cpu|cuda`` overrides; batching is the real win and works on both devices.
"""

from __future__ import annotations

import json

import math
import os

import numpy as np
import torch

from .jarvis import (_h, _inv, base_transform, camera_from_look,
                     camera_reachable, workspace_envelope, ARM_JOINTS,
                     CAMERA_ORIGIN, NECK_ORIGIN, Q_HI, Q_LO)

_DEVICE: str | None = None

_AXES = [j[2] for j in ARM_JOINTS]                       # 'x'|'y'|'z' per joint
_ORIGINS = [j[1] for j in ARM_JOINTS]                    # parent→child xyz
_Q_MID = (Q_LO + Q_HI) / 2.0


def pick_device() -> str:
    """Cached device choice: env override, else CUDA iff a real kernel runs."""
    global _DEVICE
    if _DEVICE is None:
        forced = os.environ.get("ROBOCINE_IK_DEVICE")
        if forced in ("cpu", "cuda"):
            if forced == "cuda":
                torch.zeros(2, 2, device="cuda", dtype=torch.float64).sum()
            _DEVICE = forced
            return _DEVICE
        _DEVICE = "cpu"
        if torch.cuda.is_available():
            try:
                probe = torch.zeros(4, 4, device="cuda", dtype=torch.float64)
                float((probe @ probe).sum())
                torch.cuda.synchronize()
                _DEVICE = "cuda"
            except RuntimeError:
                pass                                        # e.g. sm_120/cu126
    return _DEVICE


# --------------------------------------------------------------------------- #
# Batched primitives
# --------------------------------------------------------------------------- #
def _axis_rot_batch(axis: str, ang: torch.Tensor) -> torch.Tensor:
    """(N,3,3) principal-axis rotations for angles (N,)."""
    c, s = torch.cos(ang), torch.sin(ang)
    one, zero = torch.ones_like(ang), torch.zeros_like(ang)
    if axis == "x":
        rows = ((one, zero, zero), (zero, c, -s), (zero, s, c))
    elif axis == "y":
        rows = ((c, zero, s), (zero, one, zero), (-s, zero, c))
    else:
        rows = ((c, -s, zero), (s, c, zero), (zero, zero, one))
    return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)


def _fk_jacobian_batch(q: torch.Tensor):
    """Base-frame FK + 6×6 joint Jacobian for a batch of joint vectors.

    Mirrors ``_arm_jacobian_base``: per link the origin translation is applied
    under the PARENT rotation, then the joint rotation composes (T @ Trans(o) @
    Rot); the camera frame is link6's rotation plus the fixed CAMERA_ORIGIN
    offset.  Returns (J (N,6,6), p_cam (N,3), R_cam (N,3,3))."""
    n = q.shape[0]
    dev, dt = q.device, q.dtype
    R = torch.eye(3, dtype=dt, device=dev).expand(n, 3, 3)
    p = q.new_full((n, 3), 0.0) + q.new_tensor(NECK_ORIGIN)
    a_axis, p_axis = [], []
    for j in range(6):
        o_j = q.new_tensor(_ORIGINS[j]).reshape(1, 3, 1).expand(n, 3, 1)
        p = torch.bmm(R, o_j).squeeze(2) + p             # parent-rotation shift
        R = torch.bmm(R, _axis_rot_batch(_AXES[j], q[:, j]))
        e = torch.tensor([1.0 if k == "xyz".index(_AXES[j]) else 0.0
                          for k in range(3)], dtype=dt,
                         device=dev).reshape(1, 3, 1).expand(n, 3, 1)
        a_axis.append(torch.bmm(R, e).squeeze(2))
        p_axis.append(p)
    R_cam = R
    p_cam = torch.bmm(R, q.new_tensor(CAMERA_ORIGIN).reshape(1, 3, 1)
                      .expand(n, 3, 1)).squeeze(2) + p
    J = q.new_zeros((n, 6, 6))
    for j in range(6):
        J[:, :3, j] = torch.cross(a_axis[j], p_cam - p_axis[j], dim=1)
        J[:, 3:, j] = a_axis[j]
    return J, p_cam, R_cam


def _rot_log_batch(R: torch.Tensor) -> torch.Tensor:
    """(N,3) rotation vectors — batched mirror of ``_rot_log``, including the
    near-π axis-from-diagonal branch (and its [0,0,ang] degenerate tail)."""
    n = R.shape[0]
    c = (torch.diagonal(R, dim1=1, dim2=2).sum(dim=1) - 1.0) / 2.0
    c = c.clamp(-1.0, 1.0)
    ang = torch.acos(c)
    small = ang < 1e-9
    near_pi = ang > math.pi - 1e-6
    # generic branch: [R21-R12, R02-R20, R10-R01] / (2 sin ang) * ang
    gen = torch.stack((R[:, 2, 1] - R[:, 1, 2],
                       R[:, 0, 2] - R[:, 2, 0],
                       R[:, 1, 0] - R[:, 0, 1]), dim=1)
    safe_sin = torch.where(small, torch.ones_like(ang), torch.sin(ang))
    gen = gen / (2.0 * safe_sin).unsqueeze(1) * ang.unsqueeze(1)
    # near-π branch: axis from the diagonal, k = argmax a
    a = torch.sqrt(((torch.diagonal(R, dim1=1, dim2=2) + 1.0) / 2.0).clamp_min(0.0))
    k = a.argmax(dim=1)
    rows = torch.arange(n, device=R.device)
    a_k = a[rows, k]
    ok = a_k > 1e-9
    axis = torch.zeros_like(gen)
    axis[rows, k] = torch.where(ok, a_k, torch.ones_like(a_k))
    j, l = (k + 1) % 3, (k + 2) % 3
    denom = torch.where(ok, 2.0 * a_k, torch.ones_like(a_k))
    axis[rows, j] = torch.where(ok, R[rows, j, l] / denom,
                                torch.zeros_like(a_k))
    axis[rows, l] = torch.where(ok, R[rows, l, j] / denom,
                                torch.zeros_like(a_k))
    nrm = axis.norm(dim=1)
    degenerate = torch.zeros_like(axis) + ang.unsqueeze(1) * \
        torch.tensor([0.0, 0.0, 1.0], dtype=R.dtype, device=R.device)
    near_pi_vec = torch.where((nrm > 1e-9).unsqueeze(1),
                              axis / nrm.clamp_min(1e-300).unsqueeze(1)
                              * ang.unsqueeze(1), degenerate)
    out = torch.where(small.unsqueeze(1), torch.zeros_like(gen), gen)
    return torch.where(near_pi.unsqueeze(1), near_pi_vec, out)


# --------------------------------------------------------------------------- #
# Batched arm IK — rows = (candidate, seed), one masked Newton program
# --------------------------------------------------------------------------- #
def arm_ik_rows(p_t: torch.Tensor, R_t: torch.Tensor, seeds: torch.Tensor,
                tol_pos: float, tol_rot: float, iters_per_seed: int = 120,
                damp: float = 2e-3) -> dict:
    """Solve many independent 6-DOF arm IK problems at once.

    ``p_t``/``R_t``: target pose in each row's BASE frame (N,3)/(N,3,3);
    ``seeds``: (N,S,6) initial joint vectors.

    Mirrors ``arm_ik`` per row: damped-least-squares Newton with active-set
    joint-limit pinning (≤6 refinements), 0.35 trust region, stall>20 seed
    abandonment, convergence = pos ≤ tol_pos AND rot ≤ tol_rot at CHECK time
    (the checked q is the returned q).  Returns {ok (N,S), q (N,S,6)}."""
    n, s, _ = seeds.shape
    dev, dt = seeds.device, seeds.dtype
    q = seeds.reshape(n * s, 6).clone().clamp(
        torch.tensor(Q_LO, dtype=dt, device=dev),
        torch.tensor(Q_HI, dtype=dt, device=dev))
    q_lo = torch.tensor(Q_LO, dtype=dt, device=dev)
    q_hi = torch.tensor(Q_HI, dtype=dt, device=dev)
    eye6 = torch.eye(6, dtype=dt, device=dev)
    p_t = p_t.repeat_interleave(s, dim=0)
    R_t = R_t.repeat_interleave(s, dim=0)

    active = torch.ones(n * s, dtype=torch.bool, device=dev)
    ok = torch.zeros(n * s, dtype=torch.bool, device=dev)
    q_ok = q.clone()
    seed_best = q.new_full((n * s,), float("inf"))
    stall = torch.zeros(n * s, dtype=torch.int64, device=dev)

    for _ in range(iters_per_seed):
        run = torch.nonzero(active, as_tuple=False).squeeze(1)
        if run.numel() == 0:
            break
        J, p_cam, R_cam = _fk_jacobian_batch(q[run])
        e_p = p_t[run] - p_cam
        e_o = _rot_log_batch(torch.bmm(R_t[run], R_cam.transpose(1, 2)))
        n_p = e_p.norm(dim=1)
        n_o = e_o.norm(dim=1)
        err = torch.sqrt(n_p * n_p + n_o * n_o)

        improved = err < seed_best[run] - 1e-5
        seed_best[run] = torch.where(improved, err, seed_best[run])
        stall[run] = torch.where(improved, torch.zeros_like(stall[run]),
                                 stall[run] + 1)

        converged = (n_p <= tol_pos) & (n_o <= tol_rot)
        done = converged | (stall[run] > 20)
        ok[run[converged]] = True
        q_ok[run[converged]] = q[run][converged]
        active[run[done]] = False
        live = ~done
        if not live.any():
            continue
        run = run[live]
        qr = q[run]
        e = torch.cat((e_p[live], e_o[live]), dim=1)

        # active-set joint-limit pinning — ≤6 masked refinements per row
        m = run.numel()
        free = torch.ones((m, 6), dtype=torch.bool, device=dev)
        sub_done = torch.zeros(m, dtype=torch.bool, device=dev)
        dq = torch.zeros((m, 6), dtype=dt, device=dev)
        J_live = J[live]
        for _sub in range(6):
            work = ~sub_done
            if not work.any():
                break
            Jm = J_live * free.unsqueeze(1).to(dt)       # zero pinned columns
            A = torch.bmm(Jm, Jm.transpose(1, 2)) + (damp * damp) * eye6
            sol = torch.linalg.solve(A, e.unsqueeze(2)).squeeze(2)
            dq_new = torch.bmm(Jm.transpose(1, 2), sol.unsqueeze(2)).squeeze(2)
            at_lo = qr <= q_lo + 1e-9
            at_hi = qr >= q_hi - 1e-9
            new_pin = (at_lo & (dq_new < 0)) | (at_hi & (dq_new > 0))
            pinned = ~free
            stable = (new_pin == pinned).all(dim=1) | ~free.any(dim=1)
            dq = torch.where(work.unsqueeze(1), dq_new, dq)
            sub_done = sub_done | (work & stable)
            free = torch.where(work.unsqueeze(1), ~new_pin, free)

        step = dq.norm(dim=1)
        scale = torch.clamp(0.35 / step.clamp_min(1e-300), max=1.0)
        q[run] = (qr + dq * scale.unsqueeze(1)).clamp(q_lo, q_hi)

    return {"ok": ok.reshape(n, s), "q": q_ok.reshape(n, s, 6)}


# --------------------------------------------------------------------------- #
# Candidate enumeration — the audit pattern of camera_reachable
# --------------------------------------------------------------------------- #
def _wrap_np(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def enumerate_candidates(pos, yaw: float, pitch: float,
                         tangent_yaw: float | None, caps, sdf,
                         disc: bool = True) -> dict | None:
    """All (theta, d) base-parking candidates for one waypoint, in the exact
    order ``camera_reachable`` tries them, pre-filtered by the base-SDF and
    room-bounds checks.  Target poses in each candidate's base frame are built
    with the sequential gate's own numpy helpers (bit-identical rng digests).
    Returns {theta (C,), d (C,), p_t (C,3), R_t (C,3,3)} or None when the
    z-band rejects the pose or no candidate survives the pre-filters."""
    pos = np.asarray(pos, dtype=np.float64)
    env = workspace_envelope()
    z_lo, z_hi = env.z_band(caps.z_margin)
    if not (z_lo - 0.02 <= pos[1] <= z_hi + 0.02):
        return None

    theta0s = [tangent_yaw, _wrap_np(tangent_yaw + math.pi)] \
        if tangent_yaw is not None else [yaw]
    seen: set[float] = set()
    ordered: list[float] = []
    for th in theta0s:
        key = round(float(th), 6)
        if key not in seen:
            seen.add(key)
            ordered.append(float(th))

    if tangent_yaw is not None:                      # driving: offset 0 only
        offsets = [0.0, math.pi]                     # (|π| > 2.4 dropped below)
    else:                                            # stationary probe pattern
        offsets = sorted({0.0, round(0.0, 9), min(0.0, math.pi / 2),
                          max(0.0, -math.pi / 2), math.pi / 2, -math.pi / 2},
                         key=lambda o: abs(o - 0.0))
    offsets = [o for o in offsets if abs(o) <= 2.4]

    # arm-reach disc: the base may park anywhere on the floor disc = a
    # sphere centred on the camera (radius = max arm reach) intersected
    # with the ground plane, while holding heading = tangent (it drives
    # its own path). Enumeration is per-θ0 INTERLEAVED — tangent-rear line
    # first (legacy order, continuity-friendly), then the disc angles —
    # exactly mirroring the sequential gate's disc list per (θ0, offset).
    cand = []
    r_reach = max(caps.arm_extension)
    for theta0 in ordered:
        for off in offsets:
            cand.extend((theta0 + off, float(d), 0.0)
                        for d in caps.arm_extension)
            if disc:
                for ang in (45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0):
                    cand.append((theta0 + off, 0.75 * r_reach, float(ang)))
                    cand.append((theta0 + off, r_reach, float(ang)))
    if not cand:
        return None
    arr = np.asarray(cand, dtype=np.float64)
    theta, d = arr[:, 0], arr[:, 1]
    ang = np.deg2rad(arr[:, 2])
    ca, sa = np.cos(ang), np.sin(ang)
    rear_x, rear_z = -np.sin(theta), -np.cos(theta)
    bx = pos[0] + (rear_x * ca - rear_z * sa) * d
    bz = pos[2] + (rear_x * sa + rear_z * ca) * d

    pts = np.stack([bx, np.zeros_like(bx), bz], axis=1)
    clear = np.asarray(sdf.numpy_query(pts)) >= caps.base_clearance_margin
    m = caps.base_clearance_margin
    inside = np.ones(len(cand), dtype=bool)
    if sdf.room_bounds:
        room_idx = np.full(len(cand), -1, dtype=int)
        for ri, (lo, hi) in enumerate(sdf.room_bounds):
            hit = np.all((pts >= np.asarray(lo)) & (pts <= np.asarray(hi)),
                         axis=1)
            room_idx[hit & (room_idx < 0)] = ri
        room_idx[room_idx < 0] = 0                    # fallback: first room
        for ri in np.unique(room_idx):
            lo, hi = sdf.room_bounds[ri]
            sel = room_idx == ri
            inside[sel] = (bx[sel] >= lo[0] + m) & (bx[sel] <= hi[0] - m) & \
                          (bz[sel] >= lo[2] + m) & (bz[sel] <= hi[2] - m)
    keep = clear & inside
    if not keep.any():
        return None
    theta, bx, bz = theta[keep], bx[keep], bz[keep]

    R_look = camera_from_look(yaw, pitch)
    p_t = np.empty((len(theta), 3))
    R_t = np.empty((len(theta), 3, 3))
    for i in range(len(theta)):
        T_rel = _inv(base_transform(bx[i], bz[i], theta[i])) @ \
            _h(R_look, pos)
        p_t[i] = T_rel[:3, 3]
        R_t[i] = T_rel[:3, :3]
    return {"theta": theta, "d": d[keep], "bx": bx, "bz": bz, "p_t": p_t, "R_t": R_t}


def _seeds_for(p_t: np.ndarray, R_t: np.ndarray) -> np.ndarray:
    """Per-candidate seed set [mid, rng₁..rng₅] — the no-warm-start sequential
    list.  The rng digest is taken from the same T_rel bytes, so seed values
    match the sequential gate's first five draws bit-for-bit."""
    n = len(p_t)
    n_seeds = int(os.environ.get("ROBOCINE_IK_SEEDS", "6"))
    n_seeds = max(2, min(6, n_seeds))
    seeds = np.empty((n, n_seeds, 6))
    seeds[:, 0, :] = _Q_MID
    for i in range(n):
        T_rel = np.eye(4)
        T_rel[:3, :3] = R_t[i]
        T_rel[:3, 3] = p_t[i]
        digest = int.from_bytes(
            np.asarray(T_rel[:3, :], dtype=np.float64).tobytes()[:16], "little")
        rng = np.random.default_rng(digest)
        seeds[i, 1:, :] = rng.uniform(Q_LO, Q_HI, size=(n_seeds - 1, 6))
    return seeds


# --------------------------------------------------------------------------- #
# Batched waypoint audit
# --------------------------------------------------------------------------- #
def camera_reachable_batch(poses, yaws, pitches, tangents, sdf, caps,
                           device: str | None = None) -> dict:
    """Verdicts + representative configs for many waypoints at once.

    ``poses`` (W,3); ``yaws``/``pitches``/``tangents`` (W,) — a NaN tangent
    marks a stationary waypoint (look-yaw θ₀ pattern).  Returns
    {reachable (W,) — trusted-True verdicts (seed 0..4 successes) only,
     any_ok (W,) — incl. rng₅-only successes (callers re-check those),
     q (list) — first (candidate, seed)-order converged joint vector}."""
    device = device or pick_device()
    w = len(poses)
    reachable = np.zeros(w, dtype=bool)
    any_ok = np.zeros(w, dtype=bool)
    reps: list = [None] * w
    rep_base: list = [None] * w

    # TIERED: pass 1 runs only the tangent-rear line (legacy candidates);
    # the arm-reach disc ring runs ONLY for waypoints the rear line could
    # not certify. Candidate order WITHIN a waypoint is unchanged (rear
    # line first, then the ring) — verdict- and order-identical to the
    # single-batch enumeration, at a fraction of the IK rows.
    pending = list(range(w))
    for disc_pass in (False, True):
        if not pending:
            break
        blocks_p, blocks_R, blocks_s, w_of_row = [], [], [], []
        blocks_theta, blocks_bx, blocks_bz = [], [], []
        for i in pending:
            tang = None if (tangents is None or math.isnan(float(tangents[i]))) \
                else float(tangents[i])
            c = enumerate_candidates(poses[i], float(yaws[i]), float(pitches[i]),
                                     tang, caps, sdf, disc=disc_pass)
            if c is None:
                continue
            blocks_p.append(c["p_t"])
            blocks_R.append(c["R_t"])
            blocks_s.append(_seeds_for(c["p_t"], c["R_t"]))
            blocks_theta.append(c["theta"])
            blocks_bx.append(c["bx"])
            blocks_bz.append(c["bz"])
            w_of_row.extend([i] * len(c["theta"]))
        if not blocks_p:
            continue

        theta_all = np.concatenate(blocks_theta)
        bx_all = np.concatenate(blocks_bx)
        bz_all = np.concatenate(blocks_bz)
        p_t = torch.as_tensor(np.concatenate(blocks_p), dtype=torch.float64)
        R_t = torch.as_tensor(np.concatenate(blocks_R), dtype=torch.float64)
        seeds = torch.as_tensor(np.concatenate(blocks_s), dtype=torch.float64)
        if device == "cuda":
            p_t, R_t, seeds = p_t.cuda(), R_t.cuda(), seeds.cuda()
        res = arm_ik_rows(p_t, R_t, seeds, caps.ik_tol_pos, caps.ik_tol_rot)

        ok = res["ok"].cpu().numpy()                   # (C,6) per candidate
        q = res["q"].cpu().numpy()
        trusted = ok.copy()
        if seeds.shape[1] > 5:
            trusted[:, 5] = False                      # rng₅: sequential may lack it
        w_arr = np.asarray(w_of_row)
        nxt = []
        for i in pending:
            sel = np.nonzero(w_arr == i)[0]
            if len(sel):
                any_ok[i] = bool(any_ok[i] or ok[sel].any())
                hit = np.nonzero(trusted[sel].any(axis=1))[0]
                if len(hit):
                    ci = int(hit[0])                   # first candidate in order
                    si = int(np.argmax(ok[sel[ci]]))   # first seed in order
                    reachable[i] = True
                    reps[i] = q[sel[ci], si].copy()
                    rep_base[i] = (float(bx_all[sel[ci]]), float(bz_all[sel[ci]]),
                                   float(theta_all[sel[ci]]))
            if not reachable[i]:
                nxt.append(i)
        pending = nxt
    return {"reachable": reachable, "any_ok": any_ok, "q": reps,
            "base": rep_base}


_VERIFY_CALL_CACHE: dict = {}


def _verify_cache_key(segments, fps: float, subject) -> tuple:
    """Pose fingerprint of a verify call: the segment names/types plus each
    segment's SAMPLED camera positions and look targets at the verify fps,
    quantized to 1 cm. Verdicts are functions of the poses (not the
    params), so an optimizer that shuffles params without moving samples
    reuses the cached verdicts; any real pose change > 1 cm re-verifies —
    an order below the IK pos tolerance (3 cm), sub-threshold drift cannot
    flip a verdict."""
    import numpy as _np
    ks = []
    for seg in segments:
        try:
            th, t_abs = seg.sample_times(fps=fps)
            pos = seg.positions(th, None, subject).detach().numpy()
            look = seg.look_target(th, t_abs, subject).detach().numpy()
            pf = (pos * 100.0).round().astype(_np.int32).tobytes()
            lf = (look * 100.0).round().astype(_np.int32).tobytes()
        except Exception:
            pf = json.dumps(seg.fix, sort_keys=True, default=str)
            lf = b""
        ks.append((seg.name, seg.type, pf, lf))
    sub = getattr(subject, "subject_id", None) if subject is not None else None
    return (round(fps, 4), sub, tuple(ks))


def verify_reachable_batched(segments, subject, sdf, caps,
                             fps: float | None = None,
                             device: str | None = None) -> dict:
    """Drop-in fast ``verify_reachable_trajectory`` — identical report format,
    identical verdicts: waypoints the batch cannot certify are re-decided by
    the original sequential gate with prev-q warm-start threading.

    Results are memoized per process on the geometry fingerprint above —
    treat the returned report as read-only."""
    from ..common import constants as C
    from ..common.se3 import look_at_R, rotation_to_yaw_pitch
    fps = fps or C.DEFAULT_FPS
    _ck = _verify_cache_key(segments, fps, subject)
    if _ck in _VERIFY_CALL_CACHE:
        return _VERIFY_CALL_CACHE[_ck]

    per_seg = []
    total = 0
    for seg in segments:
        th, t_abs = seg.sample_times(fps=fps)
        pos = seg.positions(th, None, subject).numpy()
        look = seg.look_target(th, t_abs, subject).numpy()
        yawp = [rotation_to_yaw_pitch(look_at_R(p, l)) for p, l in zip(pos, look)]
        yaw = np.array([math.radians(a) for a, _ in yawp])
        pitch = np.array([math.radians(b) for _, b in yawp])
        tang = yaw.copy()
        for i in range(len(pos)):
            a = pos[max(i - 1, 0), [0, 2]]
            b = pos[min(i + 1, len(pos) - 1), [0, 2]]
            v = b - a
            if math.hypot(v[0], v[1]) > 1e-4:
                tang[i] = math.atan2(v[0], v[1])
        per_seg.append({"base": total, "pos": pos, "yaw": yaw, "pitch": pitch,
                        "tang": tang, "t_abs": t_abs})
        total += len(pos)

    verdicts = np.zeros(total, dtype=bool)
    rep_q: list = [None] * total
    if total:
        cat = lambda k: np.concatenate([s[k] for s in per_seg])   # noqa: E731
        res = camera_reachable_batch(cat("pos"), cat("yaw"), cat("pitch"),
                                     cat("tang"), sdf, caps, device=device)
        verdicts, rep_q = res["reachable"], res["q"]

    # sequential fallback — batch-unreachable (or rng₅-only) waypoints, in
    # trajectory order, warm-started exactly like the sequential gate
    okc = int(verdicts.sum())
    fallback_idx = np.nonzero(~verdicts)[0]
    if len(fallback_idx):
        starts = np.array([s["base"] for s in per_seg])
        slot_of = np.searchsorted(starts, fallback_idx, side="right") - 1
        prev_q = None
        fed = 0                                    # flat idx up to which fed
        for fi, slot in zip(fallback_idx, slot_of):
            for j in range(fed, fi):
                if verdicts[j] and rep_q[j] is not None:
                    prev_q = rep_q[j]
            fed = fi
            seg = per_seg[slot]
            i = int(fi - seg["base"])
            r = camera_reachable(seg["pos"][i], float(seg["yaw"][i]),
                                 float(seg["pitch"][i]), sdf,
                                 tangent_yaw=float(seg["tang"][i]),
                                 seed_q=prev_q, caps=caps)
            if r is not None:
                verdicts[fi] = True
                okc += 1
                prev_q = r["q"]

    violations = []
    per_segment = []
    for s_i, seg in enumerate(per_seg):
        n = len(seg["pos"])
        seg_ok = int(verdicts[seg["base"]:seg["base"] + n].sum())
        for i in range(n):
            if not verdicts[seg["base"] + i]:
                violations.append({
                    "segment": segments[s_i].name, "type": segments[s_i].type,
                    "t": round(float(seg["t_abs"][i]), 3),
                    "pos": [round(float(v), 3) for v in seg["pos"][i]]})
        per_segment.append({"segment": segments[s_i].name,
                            "type": segments[s_i].type,
                            "reachable": seg_ok, "total": int(n)})
    reasons = [f"{v['segment']} ({v['type']}) @ t={v['t']}s pos={v['pos']} "
               "不可达（底盘落点+臂 IK 无解）" for v in violations[:10]]
    report = {"status": "ok" if not violations else "failed",
              "reachable_frac": round(okc / max(total, 1), 4),
              "per_segment": per_segment,
              "violations": violations, "reasons": reasons}
    _VERIFY_CALL_CACHE[_ck] = report
    return report
