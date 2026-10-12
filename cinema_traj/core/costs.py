"""Differentiable cost terms (paper §3.4.2 eqs. 5–7 + subject extension).

All costs take torch tensors and return scalars; every term is differentiable
end-to-end w.r.t. trajectory free parameters.
"""

from __future__ import annotations

import math
import torch

from ..common.sdf import SceneSDF
from ..common.subject import SubjectTrack


def cost_sdf(pos: torch.Tensor, sdf: SceneSDF, lam: float = 5.0,
             d_safe: float = 0.2, phi: torch.Tensor | None = None) -> torch.Tensor:
    """Eq. 6: λ/|P| Σ [ρ(g) + ρ(g)²], g = d_safe − Φ(p).

    ``shortfall`` is the paper's g — how deep the camera sits inside the
    d_safe band around obstacles (0 when clear); ``penalty`` is ρ(g), the
    hinge that zeroes the term once Φ ≥ d_safe. ``phi`` may carry Φ(pos)
    already computed by the caller (the optimizer reuses the forward-pass
    values for no-grad monitoring — one SDF eval per iteration, not two)."""
    if phi is None:
        phi = sdf(pos)
    shortfall = d_safe - phi
    penalty = torch.clamp(shortfall, min=0.0)
    return lam * torch.mean(penalty + penalty * penalty)


def cost_occl(pos: torch.Tensor, target: torch.Tensor, sdf: SceneSDF,
              lam: float = 3.0, k_steps: int = 32, near_skip: float = 0.1,
              delta_occ: float = 0.05, beta: float = 20.0,
              target_mask_dist: float | None = None) -> torch.Tensor:
    """Eq. 7: transmittance-weighted occlusion along camera→target rays.

    ``hazard`` is the paper's h_m = softplus_β(δ_occ − Φ(r(s_m))) per ray
    sample; ``transmittance`` is T_m = exp(−Σ_{j<m} hazard_j) — the FIRST
    occluder dominates. Samples within ``target_mask_dist`` of the target are
    zeroed (self-occlusion guard, paper §D.5)."""
    hits = sdf.ray_march_hits(pos, target, k_steps, near_skip, target_mask_dist)
    return cost_occl_from_hits(hits, lam, delta_occ, beta)


def cost_occl_from_hits(hits: torch.Tensor, lam: float = 3.0,
                        delta_occ: float = 0.05, beta: float = 20.0) -> torch.Tensor:
    """Occlusion cost from precomputed ray samples (the optimizer marches the
    rays once and reuses them for both the cost and the early-stop count)."""
    hazard = torch.nn.functional.softplus(delta_occ - hits, beta=beta)
    prefix_excl = torch.cumsum(hazard, dim=1) - hazard     # exclusive prefix sum
    transmittance = torch.exp(-prefix_excl)
    return lam * torch.mean((transmittance * hazard).sum(dim=1))


def cost_smooth(pos: torch.Tensor, lam: float = 0.5) -> torch.Tensor:
    """Jerk (3rd finite difference) minimization, paper §D.5 λ_smooth=0.5."""
    if pos.shape[0] < 4:
        return pos.new_zeros(())
    jerk = pos[3:] - 3 * pos[2:-1] + 3 * pos[1:-2] + pos[:-3]
    return lam * torch.mean((jerk * jerk).sum(dim=-1))


def cost_drivable(pos: torch.Tensor, target: torch.Tensor, sdf: SceneSDF,
                  lam: float = 2.0, arm_yaw_window_deg: float = 85.0,
                  base_d: float = 0.40, d_safe: float = 0.20) -> torch.Tensor:
    """Differential-drive drivability of the CAMERA path (jarvis only).

    Two differentiable penalties per interior sample, both in the plan
    (x, z) plane of the Y-up world:

    1. LOOK-TANGENT dead zone — the arm compensates a look-vs-heading
       offset of at most ±arm_yaw_window_deg (85° — only a 5° margin to
       the 90° wrist branch flip where joint6 would have to reverse its
       limit face; 75° was over-conservative). A camera path whose
       tangent-vs-look angle sits near 90° cannot be driven by a diff
       base at ANY parking spot (a true orbit is exactly this dead
       zone), so the optimizer is pushed away from it:
       m = min(|Δ|, 180°−|Δ|) is the equivalent look-back angle and the
       hinge fires past the window.
    2. BASE PARKING clearance — while moving, the base may park ANYWHERE
       on the floor disc covered by a sphere of radius r_reach (= max
       arm reach) centred on the camera (sphere ∩ ground plane), holding
       heading = path tangent. The penalty is the BEST candidate's SDF
       violation (min over an azimuth×radius candidate grid): it fires
       only when the base fits nowhere on the whole disc.
       Layer split (do not conflate): this is per-point STATIC
       feasibility; the hard audit additionally requires the tangent-rear
       parking line — the base's own path (the camera path's tangent
       offset) must stay drivable — so a soft pass is not by itself an
       audit pass. d_safe defaults to the audit's base_clearance_margin
       (0.20) so the optimizer never settles in a gap the gate rejects.
    """
    if pos.shape[0] < 3:
        return pos.new_zeros(())
    step = pos[1:, [0, 2]] - pos[:-1, [0, 2]]            # (N-1, 2) planar
    # stationary samples (pan posts!) have step = 0: atan2(0, 0) forward is
    # 0 but its BACKWARD is NaN, which poisons every theta — substitute a
    # dummy direction there and mask the dead-zone term out instead
    moving = torch.linalg.norm(step, dim=-1) > 1e-6
    dummy = torch.tensor([0.0, 1.0], dtype=pos.dtype)    # +Z, masked anyway
    safe_step = torch.where(moving[:, None], step, dummy)
    tan_yaw = torch.atan2(safe_step[:, 0], safe_step[:, 1])   # yaw 0 → +Z
    tan_dir = torch.stack([torch.sin(tan_yaw), torch.cos(tan_yaw)], dim=-1)
    look = target[1:, [0, 2]] - pos[1:, [0, 2]]
    has_look = torch.linalg.norm(look, dim=-1) > 1e-9
    look_safe = torch.where(has_look[:, None], look, dummy)
    look_yaw = torch.atan2(look_safe[:, 0], look_safe[:, 1])
    delta = (look_yaw - tan_yaw + torch.pi) % (2 * torch.pi) - torch.pi
    m = torch.minimum(delta.abs(), torch.pi - delta.abs())
    window = torch.as_tensor(torch.deg2rad(torch.tensor(
        float(arm_yaw_window_deg), dtype=pos.dtype)), dtype=pos.dtype)
    deadzone = torch.where(moving,
                           torch.clamp(m - window, min=0.0),
                           torch.zeros_like(m))
    # base parking on the arm-reach disc (sphere centred on the
    # camera ∩ ground plane): min-over-candidates hinge — penalised only
    # when EVERY candidate spot on the disc is inside an obstacle
    rear = -tan_dir                                      # (N-1, 2) [x, z]
    r_reach = max(base_d, 0.60)
    cands = []
    for ang_deg in (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0):
        a = math.radians(ang_deg)
        ca, sa = math.cos(a), math.sin(a)
        dvec = torch.stack([rear[:, 0] * ca - rear[:, 1] * sa,
                            rear[:, 0] * sa + rear[:, 1] * ca], dim=-1)
        for r in (base_d, r_reach):
            p_base = pos[1:].clone()
            p_base[:, 0] = pos[1:, 0] + r * dvec[:, 0]
            p_base[:, 2] = pos[1:, 2] + r * dvec[:, 1]
            p_base[:, 1] = 0.0                             # floor plane
            cands.append(p_base)
    # ONE batched SDF evaluation over all 16·(N-1) candidate spots (was 16
    # sequential sdf() calls per iteration), split back per candidate grid
    p_all = torch.cat(cands, dim=0)
    phi_all = sdf(p_all).reshape(len(cands), -1)
    parking = torch.clamp(d_safe - phi_all, min=0.0).min(dim=0).values
    return lam * torch.mean(deadzone * deadzone
                            + parking + parking * parking)


def cost_reg(theta: torch.Tensor, theta0: torch.Tensor,
             ranges: torch.Tensor, lam: float = 0.1) -> torch.Tensor:
    """Range-normalized deviation from initial parameters."""
    if theta.numel() == 0:
        return theta.new_zeros(())
    return lam * torch.mean(torch.abs(theta - theta0) / ranges.clamp_min(1e-6))


def cost_bnd(pos: torch.Tensor, bounds: tuple, lam: float = 5.0,
             margin: float = 0.15, scale: float = 0.15) -> torch.Tensor:
    """Exponential barrier at room boundaries (paper λ_bnd=5.0)."""
    lo = pos.new_tensor(bounds[0])
    hi = pos.new_tensor(bounds[1])
    inside = torch.minimum(pos - lo, hi - pos).min(dim=-1).values   # (F,)
    barrier = torch.exp((margin - inside) / scale)
    return lam * torch.mean(barrier)


def cost_subject_clearance(pos: torch.Tensor, t_abs: torch.Tensor,
                           subject: SubjectTrack, r_min: float = 0.5,
                           lam: float = 0.3, beta: float = 10.0,
                           axes: tuple | None = None) -> torch.Tensor:
    """Extension: keep the camera at least r_min away from the subject capsule
    surface at each sampled time. Each frame is evaluated against the capsule
    pose at its own time in one vectorized batch. ``axes`` may carry endpoints
    precomputed by the optimizer because they are θ-independent."""
    if subject is None:
        return pos.new_zeros(())
    if axes is None:
        axes = subject.capsule_axes(t_abs.detach().numpy())
    axis_a_np, axis_b_np = axes
    axis_a = torch.as_tensor(axis_a_np, dtype=pos.dtype, device=pos.device)
    axis_b = torch.as_tensor(axis_b_np, dtype=pos.dtype, device=pos.device)
    axis = axis_b - axis_a
    axis_len2 = (axis * axis).sum(dim=-1).clamp_min(1e-12)
    u = torch.clamp(((pos - axis_a) * axis).sum(dim=-1) / axis_len2, 0.0, 1.0)
    proj = axis_a + u.unsqueeze(-1) * axis
    capsule_dist = (pos - proj).norm(dim=-1) - 0.30
    return lam * torch.nn.functional.softplus(
        r_min - capsule_dist, beta=beta).mean()


# --------------------------------------------------------------------------- #
# Diagnostics (no grad)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collision_stats(pos: torch.Tensor, sdf: SceneSDF) -> tuple:
    """(collision_fraction, min_clearance) — Φ < 0 counts as collision."""
    phi = sdf(pos)
    frac = float((phi < 0.0).float().mean())
    return frac, float(phi.min())


@torch.no_grad()
def occlusion_count(pos: torch.Tensor, target: torch.Tensor, sdf: SceneSDF,
                    k_steps: int = 32, near_skip: float = 0.1,
                    delta_occ: float = 0.05, beta: float = 20.0,
                    target_mask_dist: float | None = None) -> int:
    hits = sdf.ray_march_hits(pos, target, k_steps, near_skip, target_mask_dist)
    return occl_count_from_hits(hits, delta_occ, beta)


@torch.no_grad()
def occl_count_from_hits(hits: torch.Tensor, delta_occ: float = 0.05,
                         beta: float = 20.0) -> int:
    hazard = torch.nn.functional.softplus(delta_occ - hits, beta=beta)
    return int((hazard.sum(dim=1) > 1e-3).sum())
