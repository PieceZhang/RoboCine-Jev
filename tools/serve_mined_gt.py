#!/usr/bin/env python3
"""GT cleaning server for mined/pilot frames (port 8767 by default).

Large image + Range-streamed source mp4 + 5-direction toggle + per-direction /
obstacle / uncorrected filters. Atomic writeback with snapshot key
'depth_predicted' on first human correction.

  python tools/serve_mined_gt.py [json_name] [port]
"""
import json
import os
import sys
import threading
from urllib.parse import quote, urlparse, parse_qs

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from vlm_intention.common.config import load_config
from vlm_intention.common.gt import make_save_gt
from vlm_intention.common.paths import resolve_base, mp4_rel_for
from vlm_intention.common.schema import DIRS
from tools.gt_server_common import SilentHandler, send, serve_image, serve_video, run_server

_cfg = None
BASE = None
JSON_NAME = None
JSON_PATH = None
DATA = {}
SAVE_GT = None
_save_lock = threading.Lock()


def load():
    global DATA
    DATA = json.load(open(JSON_PATH, encoding="utf-8"))


def filter_keys(v):
    keys = sorted(DATA)
    if v == "all":
        return keys
    if v == "obs":
        return [k for k in keys if any(DATA[k].get(d) for d in DIRS)]
    if v == "uncorrected":
        return [k for k in keys if not DATA[k].get("corrected")]
    if v in DIRS:
        return [k for k in keys if DATA[k].get(v)]
    return keys


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>mined GT 清洗</title>
<style>
body{font-family:-apple-system,sans-serif;background:#111;color:#eee;margin:0;padding:14px}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
button{background:#2a2a2a;color:#eee;border:1px solid #444;padding:7px 14px;border-radius:6px;cursor:pointer}
button:hover{background:#3a3a3a}
.dirs{display:flex;gap:8px;margin:10px 0;flex-wrap:wrap;align-items:center}
.gbtn{font-size:20px;padding:14px 28px;border-radius:10px;min-width:70px}
.gbtn.on{background:#c0392b;border-color:#e74c3c;color:#fff}.gbtn.off{background:#2a2a2a;color:#888}
img{max-width:55%;max-height:78vh;border:1px solid #444;border-radius:6px}
video{max-width:42%;max-height:78vh;border:1px solid #444;border-radius:6px}
.row{display:flex;gap:12px;flex-wrap:wrap}
.panel{background:#1c1c1c;padding:12px;border-radius:8px;font-size:13px}
small{color:#999}.gsave{background:#27ae60;font-weight:bold}.gsave:hover{background:#2ecc71}
.kbd{background:#333;padding:1px 6px;border-radius:3px;font-family:monospace;font-size:12px}
a.fl{color:#6cf;text-decoration:none;margin-right:8px}a.fl.on{color:#fc6;font-weight:bold}
</style></head><body>
<div class="bar">
  <button onclick="go(__IM1__)">◀</button><b>__IP1__ / __N__</b><button onclick="go(__IP1__)">▶</button>
  <span>过滤：</span>__FILTERS__
  <span class="panel" style="min-width:auto">已纠正 __DONE__/__N__</span>
</div>
<div class="row">
  <img src="/img?p=__U__">
  <video src="/video?p=__VU__" controls></video>
</div>
<div class="dirs">
  <button class="gbtn __C0__" onclick="tg(0)">前<span class="kbd">1</span></button>
  <button class="gbtn __C1__" onclick="tg(1)">左<span class="kbd">2</span></button>
  <button class="gbtn __C2__" onclick="tg(2)">右<span class="kbd">3</span></button>
  <button class="gbtn __C3__" onclick="tg(3)">上<span class="kbd">4</span></button>
  <button class="gbtn __C4__" onclick="tg(4)">下<span class="kbd">5</span></button>
  <button class="gsave" onclick="sv(false)">保存<span class="kbd">S</span></button>
  <button class="gsave" onclick="sv(true)">保存+下张<span class="kbd">↵</span></button>
</div>
<div class="panel" style="max-width:600px"><div style="word-break:break-all">__NAME__</div>
<small>深度预测(快照)：__PRED__ | 模型：__MODEL__</small></div>
<script>
const DIRS=['前','左','右','上','下'];let gt={'前':__B0__,'左':__B1__,'右':__B2__,'上':__B3__,'下':__B4__};
function tg(i){const d=DIRS[i];gt[d]=!gt[d];document.querySelectorAll('.gbtn')[i].className='gbtn '+(gt[d]?'on':'off');}
function sv(next){fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({relative_path:'__RP__','前':gt['前'],'左':gt['左'],'右':gt['右'],'上':gt['上'],'下':gt['下']})}).then(r=>r.json()).then(x=>{if(!x.ok){alert('保存失败: '+x.error);return;}location='/?v=__V__&i='+(next?__IP1__:__IP0__);});}
function go(i){if(i<0)i=0;if(i>__NM1__)i=__NM1__;location='/?v=__V__&i='+i;}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const m={'1':0,'2':1,'3':2,'4':3,'5':4};if(e.key in m)tg(m[e.key]);else if(e.key==='s'||e.key==='S')sv(false);else if(e.key==='Enter')sv(true);else if(e.key==='ArrowLeft')go(__IM1__);else if(e.key==='ArrowRight')sv(true);});
</script></body></html>"""


def render_html(i, n, key, v):
    entry = DATA.get(key, {})
    pred = entry.get("depth_predicted") or {d: bool(entry.get(d)) for d in DIRS}
    state = {d: bool(entry.get(d)) for d in DIRS}
    done = sum(1 for k in DATA if DATA[k].get("corrected"))
    filt = "".join(
        f'<a class="fl {"on" if v==f else ""}" href="/?v={f}&i=0">{lab}</a>'
        for f, lab in [("obs", "有障碍"), ("uncorrected", "未纠正"), ("all", "全部")]
        + [(d, d) for d in DIRS])
    h = (HTML.replace("__IM1__", str(i-1)).replace("__IP1__", str(i+1)).replace("__IP0__", str(i))
         .replace("__NM1__", str(n-1)).replace("__N__", str(n))
         .replace("__U__", quote(key, safe=""))
         .replace("__VU__", quote(mp4_rel_for(key), safe=""))
         .replace("__RP__", json.dumps(key))
         .replace("__NAME__", key.split("/")[-1])
         .replace("__PRED__", "/".join(d for d in DIRS if pred.get(d)) or "无")
         .replace("__MODEL__", str(entry.get("model", "")).split("/")[-1])
         .replace("__V__", v).replace("__FILTERS__", filt).replace("__DONE__", str(done)))
    for j, d in enumerate(DIRS):
        h = h.replace(f"__C{j}__", "on" if state[d] else "off")
        h = h.replace(f"__B{j}__", "true" if state[d] else "false")
    return h.encode("utf-8")


class Handler(SilentHandler):
    def do_HEAD(self):
        if urlparse(self.path).path == "/video":
            serve_video(self, BASE, urlparse(self.path).query, head_only=True)
        else:
            send(self, 200)

    def do_GET(self):
        parsed = urlparse(self.path); qs = parse_qs(parsed.query)
        if parsed.path == "/img":
            serve_image(self, BASE, parsed.query)
        elif parsed.path == "/video":
            serve_video(self, BASE, parsed.query)
        elif parsed.path in ("/", "/index.html"):
            load()   # refresh from disk so progress reflects other tabs
            v = qs.get("v", ["obs"])[0]
            keys = filter_keys(v) or sorted(DATA)
            n = len(keys)
            i = max(0, min(n-1, int(qs.get("i", ["0"])[0])))
            send(self, 200, render_html(i, n, keys[i], v), "text/html; charset=utf-8")
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
        load()
        with _save_lock:
            ok, info = SAVE_GT(DATA, payload.get("relative_path"), payload)
        send(self, 200 if ok else 400,
             json.dumps({"ok": ok, "error": None if ok else info}).encode(),
             "application/json")
        print(("[SAVE]" if ok else "[ERR ]"), payload.get("relative_path"), "->", info)


def main():
    global _cfg, BASE, JSON_NAME, JSON_PATH, SAVE_GT, DATA
    from vlm_intention.common import config as cfgmod
    _cfg = cfgmod.load_config()
    BASE = resolve_base(_cfg)
    JSON_NAME = next((a for a in sys.argv[1:] if a.endswith(".json")), None) \
        or _cfg.outputs.mined_annotations
    JSON_PATH = BASE / JSON_NAME
    SAVE_GT = make_save_gt(JSON_PATH, "depth_predicted", "depth_model", _save_lock)
    DATA = json.load(open(JSON_PATH, encoding="utf-8"))
    for d in DIRS:
        n = sum(1 for v in DATA.values() if v.get(d))
        print(f"  {d}: {n}/{len(DATA)} = {n/max(len(DATA),1)*100:.1f}%")
    banner = (f"=" * 60 + f"\n  mined GT 清洗 | JSON: {JSON_NAME} | {len(DATA)} 条\n"
              f"  写回：原值快照 depth_predicted\n" + "=" * 60)
    run_server(next((int(a) for a in sys.argv[1:] if a.isdigit()), None)
               or _cfg.viewer.default_port_mined,
               Handler, JSON_NAME, banner)


if __name__ == "__main__":
    main()
