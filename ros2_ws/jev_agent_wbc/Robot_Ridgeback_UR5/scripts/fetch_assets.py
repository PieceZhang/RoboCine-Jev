#!/usr/bin/env python3
"""Download and hash the recursive official USD asset closure, without Kit/GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/"
ENTRY = "Isaac/Robots/Clearpath/RidgebackUr/ridgeback_ur5.usd"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-python", default="/home/w84384651/miniconda3/envs/env_isaaclab/bin/python")
    args = parser.parse_args()
    # The installed USD libraries can inspect layers without starting SimulationApp.
    site = Path(args.isaac_python).resolve().parents[1] / "lib/python3.11/site-packages/isaacsim"
    usd = next((site / "extscache").glob("omni.usd.libs-*"))
    if not os.environ.get("RIDGEBACK_USD_READY"):
        env = dict(os.environ, RIDGEBACK_USD_READY="1")
        env["LD_LIBRARY_PATH"] = ":".join((str(Path(args.isaac_python).resolve().parents[1] / "lib"), str(usd / "bin"), str(site / "kit"), env.get("LD_LIBRARY_PATH", "")))
        os.execve(args.isaac_python, [args.isaac_python, __file__, *sys.argv[1:]], env)
    sys.path.insert(0, str(usd))
    from pxr import Sdf
    assets = ROOT / "assets"
    manifest_path = ROOT / "ASSET_MANIFEST.json"
    previous_hashes = {}
    if manifest_path.exists():
        previous_hashes = {record["path"]:record["sha256"] for record in json.loads(manifest_path.read_text())["files"]}
    queue = [ORIGIN + ENTRY]
    seen, records, builtins = set(), [], set()
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        if not url.startswith(ORIGIN):
            raise ValueError(f"Unexpected asset outside the official pinned tree: {url}")
        relative = urllib.parse.unquote(url[len(ORIGIN):])
        path = assets / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            path.write_bytes(data)
        data = path.read_bytes()
        local_name = str(path.relative_to(ROOT))
        digest = hashlib.sha256(data).hexdigest()
        if local_name in previous_hashes and previous_hashes[local_name] != digest:
            raise ValueError(f"Local asset no longer matches its pinned SHA256: {path}")
        records.append(dict(path=local_name, url=url, bytes=len(data), sha256=digest))
        print(f"LOCAL {relative} ({len(data)} bytes)", flush=True)
        if path.suffix.lower() in (".usd", ".usda", ".usdc"):
            layer = Sdf.Layer.FindOrOpen(str(path))
            if not layer:
                raise ValueError(f"USD layer failed to parse: {path}")
            refs = set(re.findall(r"@([^@\n]*)@", layer.ExportToString()))
            for ref in refs:
                if not ref:
                    continue
                if ref in ("OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl"):
                    builtins.add(ref)
                    continue
                if ref.startswith("anon:"):
                    continue
                queue.append(urllib.parse.urljoin(url, ref))
    manifest = dict(robot="Clearpath Ridgeback + Universal Robots UR5", isaac_sim_asset_version="5.1", entry="assets/" + ENTRY, origin=ORIGIN, source_documentation="https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_robots.html", asset_license="BSD-3 (per NVIDIA asset catalog)", runtime_builtins=sorted(builtins), files=sorted(records, key=lambda item: item["path"]), downloaded_closure_complete=True)
    (ROOT / "ASSET_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(records)} local files; built-in MDL: {sorted(builtins)}")


if __name__ == "__main__":
    main()
