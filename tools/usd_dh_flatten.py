#!/usr/bin/env python3
"""usd_dh_flatten.py — flatten a DH_Characters people asset into a static,
crash-immune, simply-shaded USDA for the simview cast.

Why: DH chars render ~black in the bare RTX app and IGNORE all session-layer
material overrides (measured: root/leaf binding writes change nothing; the
Fabric material pool keeps the hashed MDL networks).  The live reference does
NOT crash (unlike the RL People chars) but is unstyleable.  So we copy the
geometry out to a brand-new file:

  * every visible Mesh: points/normals (+faceVarying st0), fvc/fvi
  * GeomSubsets per mesh (HIDDEN_* subsets dropped — they are covered skin)
  * Z-up → Y-up rotation baked into positions+normals, recentered to
    origin-at-feet
  * one flat OmniPBR-free UsdPreviewSurface per style RULE (skin/hair/eye/
    tooth/shirt/pant/shoe/…), palette taken from a small dict — see PALETTES

Meshes on SKIP_PAT are dropped wholesale (PPE: helmet/glove/headphone/vest/
overall…).  No Skeleton/SkelAnimation/BlendShape/Imageable-with-references —
guaranteed to render in the minimal headless app.

Run (boots Kit for pxr, does not render):
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
  OMNI_KIT_ACCEPT_EULA=yes python tools/usd_dh_flatten.py \
      --char <char.usd> --palette dad --out datasets/simview_assets/cast/people_dh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PALETTES = {
    "dad":   {"skin": (0.36, 0.25, 0.18), "hair": (0.03, 0.03, 0.03),
              "eye": (0.05, 0.04, 0.03), "shirt": (0.30, 0.42, 0.62),
              "pant": (0.20, 0.23, 0.30), "shoe": (0.08, 0.08, 0.09)},
    "mom":   {"skin": (0.79, 0.60, 0.47), "hair": (0.30, 0.20, 0.11),
              "eye": (0.06, 0.05, 0.04), "shirt": (0.85, 0.78, 0.68),
              "pant": (0.35, 0.30, 0.42), "shoe": (0.55, 0.52, 0.48)},
    "kid":   {"skin": (0.82, 0.62, 0.50), "hair": (0.50, 0.34, 0.18),
              "eye": (0.05, 0.04, 0.04), "shirt": (0.75, 0.20, 0.18),
              "pant": (0.25, 0.30, 0.38), "shoe": (0.85, 0.82, 0.78)},
}

# path-keyword → rule name; first match wins
RULES = [("_hidden", "drop"),                   # drop
         ("skin_mtl", "skin"), ("i2a_nvhuman", "skin"),
         ("scalp", "hair"), ("beard", "hair"), ("hair", "hair"),
         ("cornea", "eye"), ("choroid", "eye"), ("lens", "eye"),
         ("eye", "eye"),
         ("sneaker", "shoe"), ("boot", "shoe"), ("shoe", "shoe"),
         ("overall", "pant"),                   # coverall ≙ jumpsuit colour
         ("shirt", "shirt"),
         ("pant", "pant"), ("denim", "pant"),
         ]
# whole-mesh drops (PPE / work gear)
SKIP_PAT = ("helmet", "headphone", "glove", "vest_highvis", "accessories")


def classify(path: str) -> str | None:
    lp = path.lower()
    for pat, rule in RULES:
        if pat in lp:
            return rule
    return None                                  # unknown → inherit parent's


def f3(v) -> str:
    return f"({v[0]:.5f}, {v[1]:.5f}, {v[2]:.5f})"


def emit_usda(alias: str, meshes: list[dict], palette: dict) -> str:
    out = ['#usda 1.0', '(',
           f'    defaultPrim = "{alias}"',
           '    upAxis = "Y"',
           '    metersPerUnit = 1', ')', '',
           f'def Xform "{alias}"', '{', '    def Scope "Looks"', '    {']
    for rule, rgb in palette.items():
        out += [f'        def Material "{rule}"',
                '        {',
                '            token outputs:surface.connect = '
                f'</{alias}/Looks/{rule}/sr.outputs:surface>',
                '            def Shader "sr"',
                '            {',
                '                uniform token info:id = "UsdPreviewSurface"',
                f'                color3f inputs:diffuseColor = {f3(rgb)}',
                '                float inputs:roughness = 0.6',
                '                token outputs:surface', '            }',
                '        }']
    out += ['    }', '']
    for m in meshes:
        out.append(f'    def Mesh "{m["name"]}"')
        out.append('    {')
        out.append('        uniform token subdivisionScheme = "none"')
        out.append('        int[] faceVertexCounts = ['
                   + ",".join(map(str, m["fvc"])) + ']')
        out.append('        int[] faceVertexIndices = ['
                   + ",".join(map(str, m["fvi"])) + ']')
        out.append('        point3f[] points = ['
                   + ", ".join(f3(v) for v in m["pts"]) + ']')
        out.append('        normal3f[] normals = ['
                   + ", ".join(f3(v) for v in m["nrms"]) + ']')
        out.append(f'        uniform token normalsInterpolation = '
                   f'"{m["ninterp"]}"')
        if m.get("st0") is not None:
            out.append('        texCoord2f[] primvars:st0 = ['
                       + ", ".join(f"({u:.4f}, {v2:.4f})"
                                   for u, v2 in m["st0"]) + '] '
                       + f'(interpolation = "{m["st0interp"]}")')
            if m.get("st0idx") is not None:
                out.append('        int[] primvars:st0:indices = ['
                           + ",".join(map(str, m["st0idx"])) + ']')
        for sub in m["subsets"]:
            out.append(f'        def GeomSubset "{sub["name"]}"')
            out.append('        {')
            out.append('            uniform token elementType = "face"')
            out.append('            int[] indices = ['
                       + ",".join(map(str, sub["indices"])) + ']')
            out.append('            uniform token familyName = "materialBind"')
            out.append('            rel material:binding = '
                       f'</{alias}/Looks/{sub["rule"]}>')
            out.append('        }')
        if m["subsets"]:
            out.append('        rel material:binding = '
                       f'</{alias}/Looks/{m["rule"]}>')
        else:
            out.append('        rel material:binding = '
                       f'</{alias}/Looks/{m["rule"]}>')
        out.append('    }')
    out.append('}')
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", required=True)
    ap.add_argument("--palette", default="dad", choices=list(PALETTES))
    ap.add_argument("--alias", default=None)
    ap.add_argument("--out", default="datasets/simview_assets/cast/people_dh")
    args = ap.parse_args()
    alias = args.alias or Path(args.char).stem

    log = Path(f"/tmp/opencode/flatten_{alias}.log")

    def p(*a):
        print(*a, flush=True)
        with log.open("a") as fh:
            fh.write(" ".join(str(x) for x in a) + "\n")

    from isaacsim import SimulationApp
    sim = SimulationApp({"headless": True})
    try:
        from pxr import Usd, UsdGeom, UsdShade

        p(f"open {args.char}")
        stage = Usd.Stage.Open(args.char)
        meshes = []
        for prim in stage.Traverse():
            if prim.GetTypeName() != "Mesh":
                continue
            path = str(prim.GetPath())
            if any(k in path.lower() for k in SKIP_PAT):
                continue
            if UsdGeom.Imageable(prim).ComputeVisibility() == \
                    UsdGeom.Tokens.invisible:
                continue
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            if not pts:
                continue
            fvc = list(mesh.GetFaceVertexCountsAttr().Get())
            fvi = list(mesh.GetFaceVertexIndicesAttr().Get())
            nrms = list(mesh.GetNormalsAttr().Get() or [])
            ninterp = mesh.GetNormalsInterpolation()
            if not nrms:
                ninterp = "faceVarying"
                nrms = []
            pv = UsdGeom.PrimvarsAPI(prim)
            st0p = pv.GetPrimvar("st0")
            st0, st0idx, st0interp = None, None, None
            if st0p and st0p.HasAuthoredValue():
                st0 = [(float(t[0]), float(t[1])) for t in st0p.Get()]
                st0interp = st0p.GetInterpolation()
                idx = st0p.GetIndices()
                st0idx = list(idx) if idx else None

            name = path.rsplit("/", 1)[-1]
            rule = classify(path) or "skin"
            subsets = []
            drop_faces: set[int] = set()
            for ch in prim.GetChildren():
                if ch.GetTypeName() != "GeomSubset":
                    continue
                sname = ch.GetName()
                idx = list(ch.GetAttribute("indices").Get() or [])
                if "hidden" in sname.lower():
                    drop_faces.update(idx)        # covered skin: omit faces
                    continue
                sb = UsdShade.MaterialBindingAPI(ch).GetDirectBinding()
                srule = (classify(str(sb.GetMaterialPath())) or rule) \
                    if sb else rule
                if srule == "drop":
                    continue
                subsets.append({"name": sname, "indices": idx,
                                "rule": srule})
            if rule == "drop":                     # whole mesh hidden-class
                continue

            # omit hidden faces; rebuild corner-indexed arrays to match
            if drop_faces:
                keep = [i for i in range(len(fvc)) if i not in drop_faces]
                keep_set = set(keep)
                remap = {old: new for new, old in enumerate(keep)}
                spans = []
                off = 0
                for i, c in enumerate(fvc):
                    if i in keep_set:
                        spans.append((off, off + c))
                    off += c
                fvc = [fvc[i] for i in keep]
                fvi = [x for lo, hi in spans for x in fvi[lo:hi]]
                if ninterp == "faceVarying" and nrms:
                    nrms = [nrms[c] for lo, hi in spans
                            for c in range(lo, hi)]
                if st0idx is not None and st0interp == "faceVarying":
                    st0idx = [st0idx[c] for lo, hi in spans
                              for c in range(lo, hi)]
                elif st0 is not None and st0interp == "faceVarying":
                    st0 = [st0[c] for lo, hi in spans for c in range(lo, hi)]
                # subset indices tracked ORIGINAL faces → remap onto kept list
                for sub in subsets:
                    sub["indices"] = [remap[i] for i in sub["indices"]
                                      if i in remap]
            meshes.append({"name": name, "pts": [tuple(x) for x in pts],
                           "nrms": [tuple(x) for x in nrms],
                           "ninterp": ninterp, "fvc": fvc, "fvi": fvi,
                           "st0": st0, "st0idx": st0idx,
                           "st0interp": st0interp, "subsets": subsets,
                           "rule": rule})
            p(f"  keep {name}: {len(pts)} pts, {len(fvc)} faces, "
              f"{len(subsets)} subsets, rule={rule}")

        # ---- Z-up → Y-up (R_x(-90): det +1 — pure rotation, winding and
        # normals stay consistent; DO NOT flip anything) + recenter --------- #
        allp = np.concatenate([np.asarray(m["pts"], np.float64)
                               for m in meshes]) if meshes else \
            np.zeros((0, 3))
        rot = allp[:, [0, 2, 1]].copy(); rot[:, 2] *= -1.0
        c = (rot.min(0) + rot.max(0)) / 2.0
        shift = np.array([c[0], rot[:, 1].min(), c[2]])      # feet→y0, ctr xz
        for m in meshes:
            P = np.asarray(m["pts"], np.float64)
            M = P[:, [0, 2, 1]].copy(); M[:, 2] *= -1.0
            M -= shift
            m["pts"] = [tuple(v) for v in M]
            if m["nrms"]:
                N = np.asarray(m["nrms"], np.float64)
                N = N[:, [0, 2, 1]].copy(); N[:, 2] *= -1.0
                N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
                m["nrms"] = [tuple(v) for v in N]
        h = rot[:, 1].max() - rot[:, 1].min()
        p(f"flattened {len(meshes)} meshes, height={h:.3f} m")

        # face +Z: eye meshes forward-check (DHGen chars come out facing −Z
        # after the (x,z,−y) rotation when in-file forward was +Y)
        eye = [np.asarray(m["pts"], np.float64) for m in meshes
               if m["rule"] == "eye"]
        if eye:
            ez = float(np.concatenate(eye)[:, 2].mean())
            if ez < 0:
                p(f"eyes at z={ez:+.3f} → flipping to +Z forward")
                for m in meshes:
                    P = np.asarray(m["pts"], np.float64)
                    P[:, 0] *= -1.0; P[:, 2] *= -1.0
                    m["pts"] = [tuple(v) for v in P]
                    if m["nrms"]:
                        N = np.asarray(m["nrms"], np.float64)
                        N[:, 0] *= -1.0; N[:, 2] *= -1.0
                        m["nrms"] = [tuple(v) for v in N]
            else:
                p(f"eyes at z={ez:+.3f} → already +Z forward")

        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        usda = outdir / f"{alias}.usda"
        usda.write_text(emit_usda(alias, meshes, PALETTES[args.palette]),
                        encoding="utf-8")
        (outdir / f"{alias}.stats.json").write_text(json.dumps({
            "alias": alias, "src": args.char, "palette": args.palette,
            "meshes": len(meshes), "height_m": round(float(h), 3),
            "points": sum(len(m["pts"]) for m in meshes),
            "faces": sum(len(m["fvc"]) for m in meshes)}, indent=1))
        p(f"WROTE {usda}")
    finally:
        sim.close()


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------- #
# loader for OUR OWN emission (stdlib+numpy only — no pxr; simview builds run
# outside Kit).  Always returns triangles + per-face palette colours.
# --------------------------------------------------------------------------- #
import re as _re


def _bracket_body(line: str) -> str:
    return line.split("= [", 1)[1].rsplit("]", 1)[0]

def load_flattened(path: str) -> dict:
    """Parse a people_dh flattened USDA produced above.

    Returns {"verts": (N,3) float, "faces": (M,3) int64,
             "face_colors": (M,3) float, "meta": {...}}.
    """
    src = Path(path).read_text(encoding="utf-8")

    pal: dict[str, tuple] = {}
    for m in _re.finditer(
            r'def Material "(\w+)"[\s\S]*?inputs:diffuseColor = \(([^)]*)\)',
            src):
        pal[m.group(1)] = tuple(float(v) for v in m.group(2).split(","))

    meshes: list[dict] = []
    cur: dict | None = None
    in_sub = False
    for line in src.splitlines():
        ls = line.strip()
        mm = _re.match(r'def Mesh "([^"]+)"', ls)
        if mm:
            cur = {"name": mm.group(1), "subsets": []}
            meshes.append(cur)
            continue
        if cur is None:
            continue
        if ls.startswith('def GeomSubset "'):
            cur["subsets"].append({"name": ls.split('"')[1]})
            in_sub = True
            continue
        if in_sub and ls == "}":
            in_sub = False
            continue
        if ls.startswith("point3f[] points"):
            a = np.fromstring(_bracket_body(line)
                              .replace("(", " ").replace(")", " ")
                              .replace(",", " "), sep=" ")
            cur["pts"] = a.reshape(-1, 3).astype(np.float64)
        elif ls.startswith("int[] faceVertexCounts"):
            cur["fvc"] = np.fromstring(_bracket_body(line), sep=",",
                                       dtype=np.int64)
        elif ls.startswith("int[] faceVertexIndices"):
            cur["fvi"] = np.fromstring(_bracket_body(line), sep=",",
                                       dtype=np.int64)
        elif ls.startswith("int[] indices"):
            cur["subsets"][-1]["indices"] = np.fromstring(
                _bracket_body(line), sep=",", dtype=np.int64)
        elif ls.startswith("rel material:binding"):
            rule = ls.split("</")[-1].split(">")[0].rsplit("/", 1)[-1]
            if in_sub:
                cur["subsets"][-1]["rule"] = rule
            else:
                cur["rule"] = rule

    Vl, Fl, FCl = [], [], []
    voff = 0
    for e in meshes:
        P = e["pts"]
        fvc, fvi = e["fvc"], e["fvi"]
        npoly = len(fvc)
        rgb = np.asarray(pal.get(e.get("rule", "skin"),
                                 (0.5, 0.5, 0.5)))[None, :].repeat(npoly, 0)
        for sub in e["subsets"]:
            if sub.get("rule") in pal and sub.get("indices") is not None:
                rgb[sub["indices"]] = pal[sub["rule"]]
        off = 0
        for c in fvc:
            idx = fvi[off:off + c]
            for i in range(1, c - 1):                # fan-triangulate
                Fl.append((voff + int(idx[0]), voff + int(idx[i]),
                           voff + int(idx[i + 1])))
            off += c
        Vl.append(P)
        FCl.append(np.repeat(rgb, fvc - 2, axis=0))  # per-tri colours
        voff += len(P)
    V = np.concatenate(Vl, 0)
    F = np.asarray(Fl, np.int64)
    FC = np.concatenate(FCl, 0)
    h = float(V[:, 1].max() - V[:, 1].min())
    return {"verts": V, "faces": F, "face_colors": FC,
            "meta": {"height_m": h, "meshes": len(meshes),
                     "palette": pal}}


def _parse_meshes_text(src: str) -> list[dict]:
    """Shared text-array parser over our emitted USDAs (any flavour)."""
    meshes: list[dict] = []
    cur: dict | None = None
    for line in src.splitlines():
        ls = line.strip()
        mm = _re.match(r'def Mesh "([^"]+)"', ls)
        if mm:
            cur = {"name": mm.group(1)}
            meshes.append(cur)
            continue
        if cur is None:
            continue
        if ls.startswith("point3f[] points"):
            a = np.fromstring(_bracket_body(line)
                              .replace("(", " ").replace(")", " ")
                              .replace(",", " "), sep=" ")
            cur["pts"] = a.reshape(-1, 3).astype(np.float64)
        elif ls.startswith("int[] faceVertexCounts"):
            cur["fvc"] = np.fromstring(_bracket_body(line), sep=",",
                                       dtype=np.int64)
        elif ls.startswith("int[] faceVertexIndices"):
            cur["fvi"] = np.fromstring(_bracket_body(line), sep=",",
                                       dtype=np.int64)
        elif ls.startswith("texCoord2f[] primvars:st0"):
            body = _bracket_body(line)
            cur["st0interp"] = (ls.split('interpolation = "')[1]
                                .split('"')[0] if 'interpolation' in ls
                                else None)
            a = np.fromstring(body.replace("(", " ").replace(")", " ")
                              .replace(",", " "), sep=" ")
            cur["st0"] = a.reshape(-1, 2).astype(np.float64)
        elif ls.startswith("int[] primvars:st0:indices"):
            cur["st0idx"] = np.fromstring(_bracket_body(line), sep=",",
                                          dtype=np.int64)
    return meshes


def load_zip_flattened(path: str) -> dict:
    """Load a people_zip USDEF2 (atlas-textured, no subsets). Face colours are
    SAMPLED from the BaseColor atlas at each face's UV centroid — the baked
    triangle cast then keeps the char's real clothing/skin tones."""
    from PIL import Image
    src = Path(path).read_text(encoding="utf-8")
    meshes = _parse_meshes_text(src)
    texd = Path(path).parent / "textures"
    cands = sorted(texd.glob("*_BaseColor.png"))
    img = (np.asarray(Image.open(cands[0]).convert("RGB"), np.float64) / 255.0
           if cands else None)
    H, W = img.shape[:2] if img is not None else (0, 0)

    Vl, Fl, FCl = [], [], []
    voff = 0
    for e in meshes:
        P, fvc, fvi = e["pts"], e["fvc"], e["fvi"]
        npoly = len(fvc)
        if img is not None and e.get("st0") is not None:
            st0 = e["st0"]
            st0idx = e.get("st0idx")
            uv_corner = (st0[st0idx] if st0idx is not None
                         and e.get("st0interp") == "faceVarying" else
                         (st0[fvi] if e.get("st0interp") == "vertex"
                          else st0[: len(fvi)]))
            u = np.clip(uv_corner[:, 0] % 1.0, 0.0, 0.99999)
            v = np.clip(uv_corner[:, 1] % 1.0, 0.0, 0.99999)
            px = np.minimum((u * W).astype(np.int64), W - 1)
            py = np.minimum(((1.0 - v) * H).astype(np.int64), H - 1)  # GL v-flip → PIL rows
            cols = img[py, px]                       # (corners, 3)
            fc = np.zeros((npoly, 3))
            off = 0
            for fi, c in enumerate(fvc):
                fc[fi] = cols[off:off + c].mean(0)
                off += c
        else:                                        # neutral fallback
            fc = np.full((npoly, 1), 0.58) @ np.ones((1, 3))
        off = 0
        for c in fvc:                                # fan triangulate
            idx = fvi[off:off + c]
            for i in range(1, c - 1):
                Fl.append((voff + int(idx[0]), voff + int(idx[i]),
                           voff + int(idx[i + 1])))
            off += c
        Vl.append(P)
        FCl.append(np.repeat(fc, fvc - 2, axis=0))
        voff += len(P)
    V = np.concatenate(Vl, 0)
    F = np.asarray(Fl, np.int64)
    FC = np.concatenate(FCl, 0)
    return {"verts": V, "faces": F, "face_colors": FC,
            "meta": {"height_m": float(V[:, 1].max() - V[:, 1].min()),
                     "meshes": len(meshes), "atlas": bool(img is not None)}}


def load_cast_usda(path: str) -> dict:
    """format auto-detect for cast usda files (people_dh palette vs people_zip
    atlas)."""
    src = Path(path).read_text(encoding="utf-8")
    if 'def Material "atlas"' in src:
        return load_zip_flattened(path)
    return load_flattened(path)
