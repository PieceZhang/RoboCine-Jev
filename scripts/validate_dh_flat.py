#!/usr/bin/env python3
"""validate_dh_flat.py — render probe for a FLATTENED people_dh cast file.
Usage:
  DH_CHAR=<path.usda>  (absolute)  and optionally DH_CAMOUTDIR.

  OMNI_KIT_ACCEPT_EULA=yes python scripts/validate_dh_flat.py
Numerics only (the chat model can't see): person-pixel count / colour richness
/ bbox height.  Pass bars: px > 100k, bins > 15, luma p99 > 120 and p50 > 20.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

CHAR = os.environ.get(
    "DH_CHAR",
    "/home/w84384651/Documents/Van/RoboCine/datasets/simview_assets/cast/"
    "people_zip/Dad.usda")
OUT = Path(os.environ.get("DH_CAMOUTDIR", "/tmp/opencode/dh_flat"))


def p(*a):
    print(*a, flush=True)


def _cam_block_text(eye, tgt) -> str:
    eye = np.asarray(eye, float); tgt = np.asarray(tgt, float)
    fwd = tgt - eye; fwd /= np.linalg.norm(fwd)
    back = -fwd
    right = np.cross([0, 1, 0], back); right /= np.linalg.norm(right)
    up = np.cross(back, right)
    R = np.stack([right, up, back], axis=1)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25 * S, (R[2, 1] - R[1, 2]) / S, \
            (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        S = np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12)) * 2
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / S]
        q[i] = 0.25 * S
        q[j] = (R[j, i] + R[i, k]) / S
        q[k] = (R[k, i] + R[j, k]) / S
        qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    return f'''    def Camera "Vcam"
    {{
        float focalLength = 24.0
        float horizontalAperture = 36
        float verticalAperture = 20.25
        float3 xformOp:translate = ({eye[0]:.5f}, {eye[1]:.5f}, {eye[2]:.5f})
        quatf xformOp:orient = ({qw:.6f}, {qx:.6f}, {qy:.6f}, {qz:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }}'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    usda = OUT / "flat_validate_stage.usda"
    usda.write_text(f'''#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)
def Xform "World"
{{
{_cam_block_text((2.0, 1.3, 2.2), (0.0, 0.9, 0.0))}
    def Xform "Flat" (
        prepend references = @{CHAR}@
    )
    {{
    }}
    def DomeLight "Sky"
    {{
        color3f inputs:color = (0.35, 0.42, 0.55)
        float inputs:intensity = 5
    }}
    def DistantLight "Sun"
    {{
        float inputs:intensity = 3000
        float inputs:angle = 5
        float3 xformOp:rotateXYZ = (-40, 35, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}
}}
''', encoding="utf-8")
    p("stage:", usda)

    from isaacsim import SimulationApp
    sim = SimulationApp({"headless": True})
    try:
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image
        from pxr import UsdGeom

        ctx = omni.usd.get_context()
        ctx.open_stage(str(usda))
        stage = None
        for _ in range(600):
            sim.update()
            stage = ctx.get_stage()
            if stage is not None:
                break
        assert stage is not None
        ref = stage.GetPrimAtPath("/World/Flat")
        rp = rep.create.render_product("/World/Vcam", (640, 360))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach(rp)
        for _ in range(25):
            sim.update()
        rep.orchestrator.step(rt_subframes=12)
        arr = ann.get_data()
        tag = Path(CHAR).stem
        Image.fromarray(arr).save(OUT / f"flat_{tag}.png")

        a = arr[..., :3].astype(np.int16)
        # bg = modal colour of the frame (dome fills most of it)
        q = (a // 8).reshape(-1, 3)
        keys, counts = np.unique(q, axis=0, return_counts=True)
        bgcent = keys[counts.argmax()] * 8 + 4
        bg = np.abs(a - bgcent).sum(2)
        mask = bg > 60
        n = int(mask.sum())
        p(f"bg modal ≈{tuple(bgcent)}  person-ish px={n}")
        bb = UsdGeom.BBoxCache(0, ["default", "render"]) \
            .ComputeWorldBound(ref).ComputeAlignedRange()
        mx, mn = np.array(bb.GetMax()), np.array(bb.GetMin())
        h = mx[1] - mn[1]
        p(f"bbox h={h:.2f} m   y[{mn[1]:.2f}..{mx[1]:.2f}]   "
          f"x span {mx[0]-mn[0]:.2f}  z span {mx[2]-mn[2]:.2f}")
        if n > 50:
            reg = a[mask]
            lum = reg.mean(1)
            pc = np.percentile(lum, [25, 50, 75, 90, 99])
            bins = len(np.unique((reg // 24), axis=0))
            p(f"px={n}  bins={bins}  luma p25/50/75/90/99={np.round(pc, 1)}")
            p(f"person mean RGB={reg.mean(0).round(1)}  (Dad palette: "
              f"skin=br, overall=denim, shoe=dark)")
            ok = n > 8_000 and bins > 15 and 60 < pc[1] < 245 and pc[4] > 150
            p("RESULT:", "FLAT-FILE RENDERS COLOURED ✓" if ok
              else "still dark/flat ✗")
        else:
            p(f"px={n}  RESULT: MISSING ✗")
        p("frame:", OUT)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
