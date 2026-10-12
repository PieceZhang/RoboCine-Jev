#!/usr/bin/env python3
"""download_assets.py — fetch all large data assets that are NOT tracked in git.

Run on a fresh machine after cloning:

    pip install huggingface_hub
    python scripts/download_assets.py          # everything (idempotent)
    python scripts/download_assets.py --list   # show what exists / what's missing

Covers:
    - datasets/interioragent/kujiale_0021  (601 MB, SpatialVerse InteriorAgent,
      HF dataset spatialverse/InteriorAgent, scene used by the sim-viewer +
      cinema_traj/scenes/kujiale_0021.json)
    - datasets/simview_assets/Xbot.glb     (2.9 MB, three.js CC0 demo humanoid;
      bind pose is baked for the --subject-glb viewer path)
    (2026-09-10: the demo cast moved to the user-supplied NVIDIA
    Isaac_People.zip pack — KayKit/Kenney downloads retired; the
    sofa-trio figures live in datasets/simview_assets/cast/people_zip.)

Large assets deliberately RETRACKED in git (everything else): see .gitignore —
datasets/CameraMovementsRecommand_output_json + the 81 MB Replica mcmc-3dgs PLY
(Replica license is non-commercial research — see the .gitignore comment).

Replica semantic mesh (datasets/replica_v1/frl_apartment_0/.../mesh_semantic.ply)
is expected from the SceneSplat distribution — download manually if
scripts/run_full_scene_chain.sh step ⓪ is needed; the chain skips it otherwise.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (target path, description, fetcher)
XBOT_URL = ("https://raw.githubusercontent.com/mrdoob/three.js/dev/"
            "examples/models/gltf/Xbot.glb")
INTERIORAGENT = ("datasets/interioragent",
                 "InteriorAgent kujiale_0021 scene (601 MB)",
                 None)   # hf fetcher handled below
XBOT = ("datasets/simview_assets/Xbot.glb", "Xbot humanoid GLB (2.9 MB)",
        XBOT_URL)


def _fetch_xbot(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    print(f"  → {XBOT_URL}")
    with urllib.request.urlopen(XBOT_URL, timeout=300) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.rename(target)


def _fetch_interioragent(target: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("missing dep: pip install huggingface_hub  "
                 "(dataset client for hf.co)")
    snapshot_download("spatialverse/InteriorAgent", repo_type="dataset",
                      allow_patterns=["kujiale_0021/**"], local_dir=str(target))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="report, don't download")
    a = ap.parse_args()

    specs = [(INTERIORAGENT[0], INTERIORAGENT[1], INTERIORAGENT[0], "dir", _fetch_interioragent),
             (XBOT[0], XBOT[1], XBOT[0], "file", _fetch_xbot)]
    for rel, desc, target_rel, kind, fetch in specs:
        target = REPO / target_rel
        done = (target.is_dir() and any(target.iterdir())) if kind == "dir" else target.exists()
        status = "✔ exists" if done else "✘ missing"
        print(f"{status}  {rel}  — {desc}")
        if a.list or done:
            continue
        print(f"  downloading...")
        fetch(target)
        print(f"  done → {target}")


if __name__ == "__main__":
    main()
