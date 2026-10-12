"""Two-pass collision & occlusion-free optimizer (paper §3.4 + §D.5 + §F).

Pass 1 — per object-level segment, full cost (SDF + occlusion + smoothness +
regularization + boundary [+ subject clearance]), optimizing only the
per-type free-parameter subset so shot style survives (an orbit stays an
orbit). Adam lr=1.0, ≤1000 iterations, ReduceLROnPlateau(30), grad clip 1.0,
adaptive λ_sdf doubling (×2 up to 10³) when collision > 5% at plateau.

Stopping (in order): θ₀ already clean → ``bypassed`` (0 iterations — SVSDF
mid-end bypass: if the mid-end trajectory is collision-free, skip the
back-end entirely); collision AND occlusion zero → ``converged``; occlusion
non-zero but collision-free and the occluded-frame count has not improved for
``occl_plateau_iters`` → ``occl_plateau`` (occlusion often cannot reach 0 —
target-adjacent geometry); else ``max_iters``.

Pass 2 — transitional arcs rebuilt from updated endpoints and optimized over
the arc angle α only (collision-only cost). Cross-room arcs are split into
sub-arcs through door waypoints: room-connectivity BFS + passage scoring
‖x−c_p‖ + ‖c_p−x_dest‖ + 0.5·pref(p) (doorframe/archway 0 < door 2); waypoint
height clamped to the door's vertical extent (paper appendix F).

Subject handling (extension): subject segments are optimized AFTER pass 2 so
their absolute sample times are final (arc splitting changes downstream
t_starts). Transitions into a subject segment target that segment's actual
start position.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from ..common import constants as C
from ..common.scene import Scene
from ..common.sdf import SceneSDF
from ..common.subject import SubjectTrack
from . import costs
from .trajectories import ArcTransition, Segment


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class OptCfg:
    def __init__(self, d: dict | None = None):
        d = d or {}

        def g(k, default):
            return d.get(k, default)

        self.lr = float(g("lr", 1.0))
        self.max_iters = int(g("max_iters", 1000))
        self.plateau_iters = int(g("plateau_iters", 30))
        self.grad_clip = float(g("grad_clip", 1.0))
        self.lambda_sdf = float(g("lambda_sdf", 5.0))
        self.lambda_sdf_max = float(g("lambda_sdf_max", 1e3))
        self.d_safe = float(g("d_safe", 0.2))
        self.lambda_occl = float(g("lambda_occl", 3.0))
        self.ray_steps = int(g("ray_steps", 32))
        self.near_skip = float(g("near_skip", 0.1))
        self.delta_occ = float(g("delta_occ", 0.05))
        self.softplus_beta = float(g("softplus_beta", 20.0))
        self.lambda_smooth = float(g("lambda_smooth", 0.5))
        self.lambda_reg = float(g("lambda_reg", 0.1))
        self.lambda_bnd = float(g("lambda_bnd", 5.0))
        self.early_stop = bool(g("early_stop", True))
        self.collision_plateau_thresh = float(g("collision_plateau_thresh", 0.05))
        self.subject_r_min = float(g("subject_r_min", 0.5))
        self.lambda_clearance = float(g("lambda_clearance", 0.3))
        self.lambda_drivable = float(g("lambda_drivable", 2.0))
        # drivability soft-layer geometry (config-plumbable, see costs.py):
        # yaw window 85 deg = 5 deg margin to the 90 deg wrist branch flip;
        # base_d/d_safe define the parking-disc penalty.
        self.drivable_yaw_window_deg = float(g("drivable_yaw_window_deg", 85.0))
        self.drivable_base_d = float(g("drivable_base_d", 0.40))
        self.drivable_d_safe = float(g("drivable_d_safe", 0.20))
        # SVSDF-style mid-end bypass: θ₀ already collision- AND occlusion-free
        # → skip optimization entirely (reason "bypassed")
        self.bypass_if_clean = bool(g("bypass_if_clean", True))
        # occlusion often cannot reach exactly 0 (target-adjacent geometry);
        # once collision-free, stop when the occluded-frame count has not
        # improved for this many iterations (reason "occl_plateau"; 0 = off)
        self.occl_plateau_iters = int(g("occl_plateau_iters", 50))
        self.device = str(g("device", "cpu"))
        self.log_every = int(g("log_every", 0))


# --------------------------------------------------------------------------- #
# Pass 1 — single-segment optimization
# --------------------------------------------------------------------------- #
def optimize_segment(seg: Segment, scene: Scene, sdf: SceneSDF,
                     subject: SubjectTrack | None, occl_on: bool,
                     cfg: OptCfg, drivable_on: bool = False,
                     occl_sdf: SceneSDF | None = None) -> dict:
    names = [n for n in seg.optimize_names if n in seg.free]
    if not names:
        return {"segment": seg.name, "iters": 0, "reason": "no free params"}
    room_lo, room_hi = _room_bounds_for(seg, scene)
    # occlusion judge: the occlusion-only field when given (subject's own
    # person OBB dropped — the look target IS that body), else the same
    # field as collision
    osdf = occl_sdf if occl_sdf is not None else sdf

    th0 = torch.tensor([seg.free[n].value for n in names], dtype=torch.float64)
    lo = torch.tensor([seg.free[n].lo for n in names], dtype=torch.float64)
    hi = torch.tensor([seg.free[n].hi for n in names], dtype=torch.float64)
    ranges = hi - lo
    if bool((ranges <= 0).all()):
        # zero-range parameter box (e.g. the spiral construction locks
        # alpha lo==hi=0): nothing can move, so the iteration loop is
        # pure forward+backward waste. Output identical by definition.
        return {"segment": seg.name, "iters": 0,
                "reason": "zero-range params"}

    th, t_abs = seg.sample_times()
    # look targets are θ-independent for every segment type (fixed center /
    # subject key points) — evaluate once outside the loop
    target = seg.look_target(th, t_abs, subject) if occl_on else None
    mask = 0.5 * _target_half_extent(seg, scene) if occl_on else None
    # per-time capsule axes are likewise θ-independent
    subj_axes = subject.capsule_axes(t_abs.tolist()) if subject is not None else None

    # per-segment SDF pruning: restrict the batched box union to boxes near
    # the reachable AABB (extended to cover the look targets for the occlusion
    # rays; the cloud field rides along untouched). Final stats below still
    # use the FULL field.
    sdf_opt = sdf
    osdf_opt = osdf
    box = seg.reachable_aabb(subject)
    if box is not None:
        if occl_on:
            tgt_np = target.detach().numpy()
            box = (np.minimum(box[0], tgt_np.min(axis=0)),
                   np.maximum(box[1], tgt_np.max(axis=0)))
        sdf_opt = sdf.restricted_to(box)
        osdf_opt = sdf_opt if osdf is sdf else osdf.restricted_to(box)

    def _stats(thetas: torch.Tensor):
        """No-grad stats at given θ: (col_frac, occl_n) on the FULL field."""
        with torch.no_grad():
            p = seg.positions(th, {n: torch.clamp(thetas[i], lo[i], hi[i])
                                   for i, n in enumerate(names)}, subject)
            col_frac, _ = costs.collision_stats(p, sdf)
            occl_n = 0
            if occl_on:
                occl_n = costs.occlusion_count(p, target, osdf, cfg.ray_steps,
                                               cfg.near_skip, cfg.delta_occ,
                                               cfg.softplus_beta, mask)
        return col_frac, occl_n

    # mid-end bypass (SVSDF §mid-end): θ₀ already clean → nothing to optimize
    if cfg.bypass_if_clean:
        col0, occl0 = _stats(th0)
        if col0 == 0.0 and occl0 == 0:
            seg.write_back({n: th0[i] for i, n in enumerate(names)})
            return {"segment": seg.name, "iters": 0, "reason": "bypassed",
                    "lambda_sdf": cfg.lambda_sdf, "collision_rate": 0.0,
                    "occl_remaining": 0,
                    "min_clearance_m": round(_min_clearance(seg, th, names, th0, subject, sdf), 3),
                    "boxes_used": len(sdf_opt.obbs) if sdf_opt is not sdf else len(sdf.obbs)}

    theta = th0.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=cfg.plateau_iters)

    lam_sdf = cfg.lambda_sdf
    it = 0
    reason = "max_iters"
    best_occl = None                     # best (lowest) occluded-frame count seen
    best_occl_iter = 0
    occl_remaining = None
    for it in range(1, cfg.max_iters + 1):
        opt.zero_grad()
        params = {n: torch.clamp(theta[i], lo[i], hi[i]) for i, n in enumerate(names)}
        pos = seg.positions(th, params, subject)
        phi = sdf_opt(pos)
        hits = osdf_opt.ray_march_hits(pos, target, cfg.ray_steps, cfg.near_skip,
                                      mask) if occl_on else None
        total_cost = costs.cost_sdf(pos, sdf_opt, lam_sdf, cfg.d_safe, phi=phi) \
            + costs.cost_smooth(pos, cfg.lambda_smooth) \
            + costs.cost_reg(theta, th0, ranges, cfg.lambda_reg) \
            + costs.cost_bnd(pos, (room_lo, room_hi), cfg.lambda_bnd)
        if drivable_on and subject is not None:
            # differential-base drivability: keep the camera path's
            # tangent-vs-look angle out of the arm's dead zone and the
            # tangent-back base parking spot clear (jarvis only)
            target_d = seg.look_target(th, t_abs, subject)
            total_cost = total_cost + costs.cost_drivable(
                pos, target_d, sdf, lam=cfg.lambda_drivable,
                arm_yaw_window_deg=cfg.drivable_yaw_window_deg,
                base_d=cfg.drivable_base_d, d_safe=cfg.drivable_d_safe)
        if occl_on:
            total_cost = total_cost + costs.cost_occl_from_hits(
                hits, cfg.lambda_occl, cfg.delta_occ, cfg.softplus_beta)
        if subject is not None:
            total_cost = total_cost + costs.cost_subject_clearance(
                pos, t_abs, subject, cfg.subject_r_min, cfg.lambda_clearance,
                axes=subj_axes)
        total_cost.backward()
        torch.nn.utils.clip_grad_norm_([theta], cfg.grad_clip)
        opt.step()
        sched.step(float(total_cost.detach()))
        # monitoring reuses the forward-pass Φ / ray samples (pre-step θ): one
        # SDF eval + one ray march per iteration total; occlusion rays are
        # only counted when the cheap gate (no collisions) passes
        with torch.no_grad():
            col_frac = float((phi < 0.0).float().mean())
            if cfg.early_stop and col_frac == 0.0 and occl_on:
                occl_n = costs.occl_count_from_hits(hits.detach(), cfg.delta_occ,
                                                    cfg.softplus_beta)
                occl_remaining = occl_n
                if occl_n == 0:
                    # confirm at the STEPPED θ before accepting the stop
                    col2, occl2 = _stats(theta.detach())
                    if col2 == 0.0 and occl2 == 0:
                        reason = "converged"
                        break
                elif cfg.occl_plateau_iters > 0:
                    if best_occl is None or occl_n < best_occl:
                        best_occl, best_occl_iter = occl_n, it
                    elif it - best_occl_iter >= cfg.occl_plateau_iters:
                        col2, _ = _stats(theta.detach())
                        if col2 == 0.0:
                            reason = "occl_plateau"
                            break
            elif cfg.early_stop and col_frac == 0.0 and not occl_on:
                col2, _ = _stats(theta.detach())
                if col2 == 0.0:
                    reason = "converged"
                    break
        # adaptive λ_sdf: collision still >5% when lr has decayed to 10% → double
        if col_frac > cfg.collision_plateau_thresh and opt.param_groups[0]["lr"] < 0.1 * cfg.lr \
                and lam_sdf < cfg.lambda_sdf_max:
            lam_sdf = min(lam_sdf * 2.0, cfg.lambda_sdf_max)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=cfg.plateau_iters)

    with torch.no_grad():
        final = {n: torch.clamp(theta[i], lo[i], hi[i]) for i, n in enumerate(names)}
    seg.write_back(final)
    col_frac, min_clear = _min_clearance_pair(seg, th, names, final, subject, sdf)
    result = {"segment": seg.name, "iters": it, "reason": reason, "lambda_sdf": lam_sdf,
              "collision_rate": round(col_frac, 4), "min_clearance_m": round(min_clear, 3),
              "boxes_used": len(sdf_opt.obbs) if sdf_opt is not sdf else len(sdf.obbs)}
    if occl_on and occl_remaining is not None:
        result["occl_remaining"] = occl_remaining
    return result


def _min_clearance(seg, th, names, thetas, subject, sdf) -> float:
    with torch.no_grad():
        params = {n: torch.clamp(thetas[i], seg.free[n].lo, seg.free[n].hi)
                  for i, n in enumerate(names)}
        pos = seg.positions(th, params, subject)
        return float(sdf(pos).min())


def _min_clearance_pair(seg, th, names, params, subject, sdf) -> tuple:
    with torch.no_grad():
        pos = seg.positions(th, params, subject)
        return costs.collision_stats(pos, sdf)


def _room_bounds_for(seg: Segment, scene: Scene):
    for r in scene.rooms:
        if r.id == seg.room:
            return r.bounds_min - 0.05, r.bounds_max + 0.05
    lo = np.min([r.bounds_min for r in scene.rooms], axis=0)
    hi = np.max([r.bounds_max for r in scene.rooms], axis=0)
    return lo, hi


def _target_half_extent(seg: Segment, scene: Scene) -> float:
    if seg.target_mode == "object":
        obj = scene.object_by_id(seg.target_id)
        return obj.obb.max_half_extent() if obj else 0.5
    return 0.35                                  # subject head/chest guard radius


# --------------------------------------------------------------------------- #
# Door routing (paper appendix F)
# --------------------------------------------------------------------------- #
def room_connectivity(scene: Scene) -> dict:
    graph: dict[str, list] = {r.id: [] for r in scene.rooms}
    for d in scene.doors:
        if len(d.rooms) < 2:
            continue
        a, b = d.rooms[0], d.rooms[1]
        graph.setdefault(a, []).append((b, d))
        graph.setdefault(b, []).append((a, d))
    return graph


def door_route(scene: Scene, x_curr: np.ndarray, x_dest: np.ndarray,
               room_a: str, room_b: str) -> list[np.ndarray]:
    """BFS room path → best door per hop → waypoint positions (height clamped
    to the door's vertical extent)."""
    if room_a == room_b:
        return []
    graph = room_connectivity(scene)
    prev: dict[str, tuple[str, object] | None] = {room_a: None}
    q = deque([room_a])
    while q:
        cur = q.popleft()
        if cur == room_b:
            break
        for nxt, _door in graph.get(cur, []):
            if nxt not in prev:
                prev[nxt] = (cur, _door)
                q.append(nxt)
    if room_b not in prev:
        return []                                 # no connectivity: direct arc
    # walk back, collecting (room_entered, door_used)
    hops = []
    cur = room_b
    while prev[cur] is not None:
        src, door = prev[cur]
        hops.append((src, cur, door))
        cur = src
    hops.reverse()
    waypoints = []
    pos = np.asarray(x_curr, dtype=np.float64)
    for i, (src, dst, door) in enumerate(hops):
        c_p = door.obb.center.copy()
        pref = C.DOOR_PREF.get(door.kind, 2.0)
        # pick the best door among all doors connecting this same hop
        best, best_score = door, None
        for cand in scene.doors:
            if set(cand.rooms) == {src, dst}:
                s = (np.linalg.norm(pos - cand.obb.center)
                     + np.linalg.norm(cand.obb.center - x_dest) + 0.5 * pref)
                if best_score is None or s < best_score:
                    best, best_score = cand, s
        c_p = best.obb.center.copy()
        # clamp height into the door's vertical extent
        target_y = 0.5 * (float(pos[1]) + float(x_dest[1]))
        half_y = float(best.obb.extents[1]) / 2.0
        c_p[1] = float(np.clip(target_y, c_p[1] - half_y + 0.25, c_p[1] + half_y - 0.25))
        waypoints.append(c_p)
        pos = c_p
    return waypoints


# --------------------------------------------------------------------------- #
# Two-pass trajectory optimization
# --------------------------------------------------------------------------- #
def optimize_trajectory(segments: list, scene: Scene, sdf: SceneSDF,
                        subject: SubjectTrack | None, cfg: OptCfg | None = None,
                        arc_refine=None, drivable_on: bool = False,
                        occl_sdf: SceneSDF | None = None) -> tuple:
    cfg = cfg or OptCfg()
    log: list = []

    is_arc = lambda s: s.type == "arc"            # noqa: E731
    is_subject_seg = lambda s: s.target_mode.startswith("subject")   # noqa: E731

    # -- pass 1a: object-level static-target segments ----------------------- #
    for seg in segments:
        if not is_arc(seg) and not is_subject_seg(seg):
            r = optimize_segment(seg, scene, sdf, subject, True, cfg, drivable_on, occl_sdf)
            log.append({"pass": 1, **r})

    # -- split planning for cross-room arcs (changes durations) ------------- #
    new_segments: list = []
    for i, seg in enumerate(segments):
        if not is_arc(seg):
            new_segments.append(seg)
            continue
        prev_seg = _nearest(new_segments, back=True, predicate=lambda s: not is_arc(s))
        next_seg = _nearest(segments[i + 1:], back=False, predicate=lambda s: not is_arc(s))
        room_a = prev_seg.room if prev_seg else ""
        room_b = next_seg.room if next_seg else room_a
        if room_a and room_b and room_a != room_b and prev_seg is not None and next_seg is not None:
            p_from = prev_seg.endpoints(subject)[1]
            p_to = _segment_start(next_seg, subject)
            wps = door_route(scene, p_from, p_to, room_a, room_b)
            n = len(wps) + 1
            sub_dur = max(seg.duration / n, 1.0)
            for j in range(n):
                sub = ArcTransition(f"{seg.name}_sub{j}", sub_dur, 0.0, room_b,
                                    seg.target_mode, seg.target_id)
                if "center" in seg.fix:
                    sub.fix["center"] = seg.fix["center"]
                new_segments.append(sub)
            log.append({"pass": 2, "segment": seg.name, "split_into": n,
                        "waypoints": [w.tolist() for w in wps]})
        else:
            new_segments.append(seg)

    # -- re-accumulate absolute times (splits changed durations) ------------ #
    t = new_segments[0].t_start if new_segments else 0.0
    for seg in new_segments:
        seg.t_start = t
        t += seg.duration

    # -- pass 2: rebuild + optimize arcs ------------------------------------ #
    for i, seg in enumerate(new_segments):
        if not is_arc(seg):
            continue
        prev_seg = _nearest(new_segments[:i], back=True, predicate=lambda s: not is_arc(s))
        next_seg = _nearest(new_segments[i + 1:], back=False, predicate=lambda s: not is_arc(s))
        p_s = prev_seg.endpoints(subject)[1] if prev_seg else seg.fix.get("p_s", np.zeros(3))
        if isinstance(p_s, list):
            p_s = np.asarray(p_s)
        if next_seg is not None:
            p_e = _segment_start(next_seg, subject)
        else:
            p_e = seg.fix.get("p_e", p_s + np.array([1.0, 0.0, 0.0]))
            p_e = np.asarray(p_e) if not isinstance(p_e, list) else np.asarray(p_e)
        seg.rebuild(np.asarray(p_s), np.asarray(p_e))
        r = optimize_segment(seg, scene, sdf, subject, False, cfg, drivable_on)  # collision-only
        # Adam on the single α can stall (kujiale arc_1: collision 0.29 after
        # 1000 iters) — a discrete α grid evaluated on (collision, platform
        # extras) is a cheap robust fallback. ``arc_refine`` (optional) scores
        # one arc with its α already set: {"collision": frac, "reachable": frac}.
        if r.get("collision_rate", 0.0) > 0.0 and arc_refine is not None:
            grid = _arc_alpha_grid_search(seg, arc_refine,
                                          current=float(seg.free["alpha_deg"].value))
            if grid is not None:
                r["alpha_grid"] = grid
                r["collision_rate"] = grid["collision"]
        log.append({"pass": 2, **r})

    # -- pass 1b: subject segments (absolute times now final) ---------------- #
    for seg in new_segments:
        if not is_arc(seg) and is_subject_seg(seg):
            r = optimize_segment(seg, scene, sdf, subject, True, cfg, drivable_on, occl_sdf)
            log.append({"pass": "1b", **r})

    return new_segments, log


def _arc_alpha_grid_search(seg, evaluate, current: float,
                           grid=(-80.0, -60.0, -40.0, -20.0, 0.0, 20.0, 40.0,
                                 60.0, 80.0)) -> dict | None:
    """Discrete α fallback for a stalled arc: rebuild is NOT needed (p_s/p_e/
    chord/n_hat are α-independent) — just set α, score, keep the winner.

    Score key: fewer collisions, then higher reachability (platform-supplied),
    then smaller |α|. Includes the Adam-optimized α as a candidate. Returns
    the winning evaluation dict (with the chosen ``alpha_deg``) or None."""
    from .trajectories import Param

    cands = sorted(set(round(float(a), 3) for a in grid) |
                   {round(float(current), 3)})
    best = None
    for a in cands:
        seg.free["alpha_deg"] = Param(float(a), -89.0, 89.0)
        try:
            ev = evaluate(seg)
        except Exception:                     # pragma: no cover — a bad α
            continue
        if not ev:
            continue
        key = (float(ev.get("collision", 1.0)),
               -float(ev.get("reachable", 0.0)),
               abs(a))
        if best is None or key < best[0]:
            best = (key, a, ev)
    if best is None:
        return None
    seg.free["alpha_deg"] = Param(float(best[1]), -89.0, 89.0)
    out = dict(best[2])
    out["alpha_deg"] = best[1]
    return out


def _nearest(segments, back: bool, predicate):
    """Nearest preceding (back=True, searching a prefix list) or following
    non-arc segment."""
    if back:
        for s in reversed(segments):
            if predicate(s):
                return s
        return None
    for s in segments:
        if predicate(s):
            return s
    return None


def _segment_start(seg: Segment, subject) -> np.ndarray:
    """Start position of a segment (subject segments evaluate at their final
    absolute start time)."""
    th = torch.tensor([0.0], dtype=torch.float64)
    if seg.target_mode.startswith("subject") and subject is not None:
        return seg.positions(th, None, subject).numpy()[0]
    return seg.positions(th, None, subject).numpy()[0]
