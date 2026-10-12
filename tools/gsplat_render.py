#!/usr/bin/env python3
"""gsplat_render.py — TRUE 3DGS (Gaussian-splat) POV render of RoboCine trajectories.

Isaac Sim 5.1 pip ships no Gaussian splatting kernel, so this tool is a thin
adapter over the Nerfstudio `gsplat` CUDA renderer (install once:
`conda install -c nvidia -c conda-forge cuda-toolkit=12.8.* && pip install gsplat`;
kernels JIT-compile ~1 min on first use). It is standalone — no Isaac/Kit needed.

Design: PLY → torch tensors per the Nerfstudio/3DGS spec (scales=exp, opacities=
sigmoid, quats=normalize, colors=0.5+SH_C0·f_dc), traj_final frames → per-frame
world→camera viewmat in OpenCV convention (same math as our USD pipeline, so the
rendered shot matches the sim viewer position-by-position).

Usage (from repo root, env_isaaclab):

    python tools/gsplat_render.py --traj cinema_traj_out/frl_apartment_0_autosem_full/traj_final.json
    # → cinema_traj_out/<scene>/simview/frames_splat/{ShotCam_0000...}, ShotCam_splat.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

SH_C0 = 0.2820547913868781                # 3DGS degree-0 SH constant (view.py parity)
_DEFAULT_FOV = 55.0                        # same fallback as simview_isaac.py


# --------------------------------------------------------------------------- #
# 3DGS PLY → gsplat tensors
# --------------------------------------------------------------------------- #
def load_3dgs_ply(path: str | Path, up: str = "z",
                  shift=(0.0, 0.0, 0.0)) -> dict[str, np.ndarray]:
    """Standard 3DGS-ply (x,y,z, f_dc_0..2, opacity, scale_0..2, rot_0..3) →
    numpy fields. `up`/`shift` mirror the scene-file provenance (Replica
    SceneSplat exports are Z-up with ground in XY; y shift matches the
    hand-refined scene JSON's "floor=0")."""
    path = Path(path)
    with open(path, "rb") as fh:
        header = b""
        while not header.endswith((b"end_header\n", b"end_header\r\n")):
            byte = fh.read(1)
            if not byte:
                raise ValueError(f"{path}: truncated PLY header")
            header += byte
        h = header.decode("latin1")
        if "binary_little_endian" in h:
            endian = "<"
        elif "binary_big_endian" in h:
            endian = ">"
        else:
            raise ValueError(f"{path}: ASCII PLY not expected for 3DGS")
        props, n, in_v = [], 0, False
        for line in h.splitlines():
            p = line.split()
            if not p:
                continue
            if p[0] == "element":
                in_v = p[1] == "vertex"
                if in_v:
                    n = int(p[2])
            elif p[0] == "property" and in_v:
                props.append(p[-1])
        names = props  # keep simple -- 3DGS fields are all float32
        dt = np.dtype([(nm, endian + "f4") for nm in names])
        rec = np.frombuffer(fh.read(n * dt.itemsize), dtype=dt, count=n)

    required = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
    missing = [k for k in required if k not in names]
    if missing:
        raise ValueError(f"{path}: missing 3DGS props {missing}; got {names}")

    means = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)
    if up in ("z", "+z", "Z"):                       # right-handed (x,y,z)->(x,z,-y)
        means = np.stack([means[:, 0], means[:, 2], -means[:, 1]], axis=1)
    means += np.asarray(shift, dtype=np.float64)

    f_dc = np.stack([rec[f"f_dc_{i}"] for i in range(3)], axis=1)
    colors = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)   # DC term only (no SH here)

    return {
        "means": means.astype(np.float32),
        "quats": np.stack([rec[f"rot_{i}"] for i in range(4)], axis=1),      # raw (normalize later)
        "scales": np.stack([rec[f"scale_{i}"] for i in range(3)], axis=1),   # log-space
        "opacities": rec["opacity"],                                          # logits
        "colors": colors.astype(np.float32),
    }


# --------------------------------------------------------------------------- #
# camera: traj frame → world→camera viewmat (OpenCV: x-right, y-down, z-fwd)
# --------------------------------------------------------------------------- #
def viewmat_from_pos_look(pos, look, world_up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Same physical look-at convention as simview_isaac._camera_basis
    (USD: cam looks -Z local) then handedness-flipped to OpenCV/gsplat
    (looks +Z, y-down): R_cv = [x_right, y_down, z_fwd]."""
    pos = np.asarray(pos, np.float64).reshape(3)
    look = np.asarray(look, np.float64).reshape(3)
    f = look - pos
    n = np.linalg.norm(f)
    if n < 1e-9:
        raise ValueError("pos == look")
    f = f / n
    up = np.asarray(world_up, np.float64)
    if np.linalg.norm(np.cross(f, up)) < 1e-6:
        up = up + np.array([1e-3, 0.0, 0.0])
        up /= np.linalg.norm(up)
    r = np.cross(f, up); r = r / np.linalg.norm(r)     # f×up = image-right (gluLookAt)
    u = np.cross(r, f)                                  # up of camera in world space
    c2w = np.eye(4)
    c2w[:3, 0] = r              # x right
    c2w[:3, 1] = -u             # y DOWN (image rows top-first; matches Kit render flip)
    c2w[:3, 2] = f              # z forward    — columns [r, −u, f] form a proper rotation
    c2w[:3, 3] = pos
    return np.linalg.inv(c2w)                           # world → camera


def intrinsics(height: int, width: int, vfov_deg: float) -> np.ndarray:
    f = (height / 2.0) / np.tan(np.radians(vfov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]],
                    dtype=np.float32)


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def run(traj_path: Path, scene_path: Path | None, out_dir: Path | None,
        width: int, height: int, downsample: int, video_fps: float) -> None:
    import torch                                   # GPU gate imports here
    from gsplat import rasterization

    traj = json.loads(Path(traj_path).read_text())
    sid = traj.get("scene_id", "scene")
    scene_path = scene_path or Path(_REPO_ROOT) / "cinema_traj" / "scenes" / f"{sid}.json"
    scene = json.loads(scene_path.read_text())
    prov = scene.get("provenance", {})
    src = prov.get("source_map")
    if not src:
        raise ValueError(f"{scene_path}: no provenance.source_map (need 3DGS ply)")
    ply = Path(src)
    if not ply.exists():
        ply = Path(_REPO_ROOT) / src
    if not ply.exists():
        raise FileNotFoundError(ply)
    if "map_up" not in prov:
        raise ValueError(f"{scene_path}: provenance.map_up required with source_map ")
    up = str(prov["map_up"])
    shift = prov.get("map_shift", [0, 0, 0])

    print(f"📦 3DGS: {ply.name} (up={up} shift={shift})")
    g = load_3dgs_ply(ply, up=up, shift=shift)
    if downsample > 1:                             # quick passes
        sel = slice(None, None, downsample)
        g = {k: v[sel] for k, v in g.items()}
    print(f"   splats: {len(g['means'])}")

    means = torch.tensor(g["means"], device="cuda")
    quats = torch.nn.functional.normalize(torch.tensor(g["quats"], device="cuda"), dim=-1)
    scales = torch.exp(torch.tensor(g["scales"], device="cuda"))
    opacit = torch.sigmoid(torch.tensor(g["opacities"], device="cuda"))
    colors = torch.tensor(g["colors"], device="cuda")

    frames = traj["frames"]
    if not frames:
        raise ValueError("traj_final.json has no frames")
    t0 = frames[0]["t"]

    out_dir = out_dir or Path(traj_path).parent / "simview" / "frames_splat"
    out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image                            # noqa: some envs ship PIL w/o cv2
    have_cv2 = True
    try:
        import cv2
    except ImportError:
        have_cv2 = False
    writer = None

    print(f"🎥 {len(frames)} frames → {out_dir} ({width}×{height})")
    t_start = time.time()
    for i, f in enumerate(frames):
        pos, look = f["pos"], f["look"]
        fov = f.get("fov_deg") or _DEFAULT_FOV
        vm = torch.tensor(viewmat_from_pos_look(pos, look), device="cuda",
                          dtype=torch.float32)[None]
        ks = torch.tensor(intrinsics(height, width, fov), device="cuda",
                          dtype=torch.float32)[None]
        renders, alphas, _ = rasterization(
            means, quats, scales, opacit, colors, vm, ks,
            width, height,
            near_plane=0.05, far_plane=500.0,
            packed=False, sh_degree=None,
        )
        img = renders[0].detach().cpu().clamp(0, 1).numpy()
        png = (img * 255).astype(np.uint8)
        Image.fromarray(png).save(out_dir / f"ShotCam_{i:04d}.png")
        if have_cv2 and video_fps > 0:
            if writer is None:
                writer = cv2.VideoWriter(str(out_dir / "ShotCam_splat.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         video_fps, (width, height))
            writer.write(cv2.cvtColor(png, cv2.COLOR_RGB2BGR))
        if i == 0:
            cov = (alphas[0] > 0.5).float().mean().item()
            print(f"   frame0 coverage(α>0.5)={cov:.0%} bright={img.mean():.2f}")
            if img.mean() < 0.01 and cov < 0.02:
                raise RuntimeError("black frame — bad viewmat/frame convention "
                                   "(run tests/test_gsplat_adapter.py first)")
    if writer is not None:
        writer.release()
    print(f"✅ done {len(frames)}f in {time.time()-t_start:.0f}s"
          f" | {out_dir}/ShotCam_splat.mp4")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traj", required=True, help="traj_final.json")
    ap.add_argument("--scene", help="scene JSON (default: cinema_traj/scenes/<scene_id>.json)")
    ap.add_argument("--out", help="output dir (default: <traj_dir>/simview/frames_splat)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--downsample", type=int, default=1,
                    help="keep every Nth splat (fast previews)")
    ap.add_argument("--video-fps", type=float, default=10.0,
                    help="0 = PNGs only")
    a = ap.parse_args()
    run(Path(a.traj), Path(a.scene) if a.scene else None,
        Path(a.out) if a.out else None, a.width, a.height, a.downsample, a.video_fps)


if __name__ == "__main__":
    main()
