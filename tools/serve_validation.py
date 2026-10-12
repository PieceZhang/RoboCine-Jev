#!/usr/bin/env python3
"""GT audit server for the VLM-era labels (port 8765 by default).

Serves a fixed random sample of frames (image + Range-streamed source mp4) via
the make_validation_interface HTML; submissions write back to the loaded JSON
with snapshot key 'vlm_predicted' on first human correction.

  python tools/serve_validation.py [N] [seed] [port] [json_name]
"""
import json
import os
import sys
import threading
from urllib.parse import urlparse, parse_qs

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from vlm_intention.common.config import load_config
from vlm_intention.common.gt import make_save_gt
from vlm_intention.common.paths import resolve_base
from vlm_intention.common.schema import DIRS
from tools import make_validation_interface as gen
from tools.gt_server_common import SilentHandler, send, serve_image, serve_video, run_server

_cfg = None
BASE = None
JSON_PATH = None
DATA = {}
SAMPLE = []
SEED = 42
SAVE_GT = None
_save_lock = threading.Lock()


class Handler(SilentHandler):
    def do_HEAD(self):
        if urlparse(self.path).path == "/video":
            serve_video(self, BASE, urlparse(self.path).query, head_only=True)
        else:
            send(self, 200)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/img":
            serve_image(self, BASE, parsed.query)
        elif parsed.path == "/video":
            serve_video(self, BASE, parsed.query)
        elif parsed.path in ("/", "/index.html"):
            vd = gen.build_validation_data(SAMPLE, DATA)
            html = gen.build_html(BASE, SAMPLE, DATA, vd, SEED, len(DATA), served=True)
            send(self, 200, html.encode("utf-8"), "text/html; charset=utf-8")
        else:
            send(self, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/save":
            send(self, 404); return
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        try:
            payload = json.loads(raw.decode() or "{}")
        except Exception as e:
            send(self, 400, json.dumps({"ok": False, "error": str(e)}).encode()); return
        with _save_lock:
            ok, info = SAVE_GT(DATA, payload.get("relative_path"), payload)
        send(self, 200 if ok else 400,
             json.dumps({"ok": ok, "error": None if ok else info}).encode(),
             "application/json")
        print(("[SAVE]" if ok else "[ERR ]"), payload.get("relative_path"), "->", info)


def main():
    global _cfg, BASE, JSON_PATH, DATA, SAMPLE, SEED, SAVE_GT
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    n = int(args[0]) if len(args) > 0 and args[0].isdigit() else 15
    seed = int(args[1]) if len(args) > 1 and args[1].isdigit() else 42
    _cfg = load_config()
    port = int(args[2]) if len(args) > 2 and args[2].isdigit() else _cfg.viewer.default_port_validation
    json_name = next((a for a in args if a.endswith(".json")), None)
    BASE = resolve_base(_cfg)
    JSON_PATH = BASE / (json_name or _cfg.outputs.vlm_annotations)
    SAVE_GT = make_save_gt(JSON_PATH, "vlm_predicted", "vlm_model", _save_lock)
    DATA = json.load(open(JSON_PATH, encoding="utf-8"))
    SEED = seed
    SAMPLE = gen.pick_sample(DATA, BASE, n, seed)
    has_obs = sum(1 for k in SAMPLE if any(DATA[k].get(d) for d in DIRS))
    banner = (f"=" * 60 + f"\n  VLM GT 抽查服务 | JSON: {JSON_PATH.name}\n"
              f"  抽样 {len(SAMPLE)} (含障碍 {has_obs}), seed={seed}\n"
              f"  写回：原值快照 vlm_predicted\n" + "=" * 60)
    run_server(port, Handler, JSON_PATH.name, banner)


if __name__ == "__main__":
    main()
