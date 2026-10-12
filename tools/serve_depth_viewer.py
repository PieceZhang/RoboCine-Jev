#!/usr/bin/env python3
"""Depth-map viewer + GT writeback (port 8766 by default).

Original image + colorized depth map (5-region overlay, hover for metres) +
per-region near-fraction sidebar + 5-direction GT toggles that write back to
the loaded JSON (snapshot key 'predicted' on first human correction).

  python tools/serve_depth_viewer.py [json_name] [port] [--cpu]
"""
import json
import os
import sys
import threading
from pathlib import Path
from urllib.parse import quote, urlparse, parse_qs

# Bootstrap repo root so `tools` + `vlm_intention` packages resolve when run
# directly (python tools/serve_depth_viewer.py) as well as via -m.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vlm_intention.common.config import load_config
from vlm_intention.common.depth_model import DepthObstacleAnnotator
from vlm_intention.common.geometry import region_depths
from vlm_intention.common.gt import make_save_gt
from vlm_intention.common.paths import resolve_base
from vlm_intention.common.schema import DIRS
from vlm_intention.common.thresholds import DIST_THRESH, FRAC_THRESH, DEPTH_MIN, DEPTH_MAX, VERT_FRAC
from tools.gt_server_common import SilentHandler, send, serve_image, serve_video, run_server

_annotator = None
_save_lock = threading.Lock()
DATA = {}                       # loaded annotations (key -> entry)
ALL_KEYS, OBS_KEYS = [], []
JSON_PATH = None
SAVE_GT = None
_cfg = None
BASE = None
DEPTH_CACHE = None


def get_annotator():
    global _annotator
    if _annotator is None:
        # --cpu forces CPU; else honor cfg.models.depth.device (default cpu — MPS
        # deadlocks on this model).
        dev = "cpu" if "--cpu" in sys.argv else None
        print(f"🤖 加载深度模型（device={dev or _cfg.models.depth.device}）...")
        _annotator = DepthObstacleAnnotator(str(BASE), read_only=True, device=dev, cfg=_cfg)
    return _annotator


def _env_of(key):
    return get_annotator().classify_env(Path(key))


def get_depth_array(key):
    env = _env_of(key)
    npy = DEPTH_CACHE / (key + f".{env}.npy")
    if npy.exists():
        return np.load(npy)
    depth = get_annotator().infer_depth(BASE / key, env).astype(np.float32)
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, depth)
    return depth


def region_close_frac(depth):
    regions = region_depths(depth, vert_frac=VERT_FRAC)
    out = {}
    for d in DIRS:
        arr = regions[d]
        valid = np.isfinite(arr) & (arr >= DEPTH_MIN) & (arr <= DEPTH_MAX)
        frac = float(((arr < DIST_THRESH[d]) & valid).sum()) / float(valid.sum()) if valid.sum() else 0.0
        out[d] = (frac, DIST_THRESH[d])
    return out


def render_depth_png(key):
    env = _env_of(key)
    cache = DEPTH_CACHE / (key + f".{env}.png")
    if cache.exists():
        return cache.read_bytes()
    depth = get_depth_array(key)
    h, w = depth.shape
    finite = np.isfinite(depth)
    hi = min(10.0, float(np.percentile(depth[finite], 98))) if finite.any() else 10.0
    norm = np.clip((depth - DEPTH_MIN) / max(hi - DEPTH_MIN, 1e-6), 0, 1)
    inv = ((1.0 - norm) * 255).astype(np.uint8)
    color = cv2.applyColorMap(inv, cv2.COLORMAP_INFERNO)
    w3 = w // 3; hb = int(h * VERT_FRAC)
    for x in (w3, 2 * w3):
        cv2.line(color, (x, 0), (x, h), (220, 220, 220), 1)
    for y in (hb, h - hb):
        cv2.line(color, (0, y), (w, y), (220, 220, 220), 1)
    pil = Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil, "RGBA")
    try:
        font = ImageFont.truetype(_cfg.viewer.cjk_font, max(16, h // 40))
    except Exception:
        font = ImageFont.load_default()
    ym = h // 2
    labels = {"上": (w // 2, hb // 2), "下": (w // 2, h - hb // 2),
              "左": (w3 // 2, ym), "前": (w // 2, ym), "右": ((2 * w3 + w) // 2, ym)}
    for txt, (cx, cy) in labels.items():
        b = draw.textbbox((0, 0), txt, font=font)
        draw.text((cx - (b[2]-b[0])//2 - b[0], cy - (b[3]-b[1])//2 - b[1]), txt, font=font,
                  fill=(255, 255, 255, 210), stroke_width=2, stroke_fill=(0, 0, 0, 210))
    color = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cache.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", color)
    cache.write_bytes(buf.tobytes())
    return buf.tobytes()


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>深度图浏览器</title>
<style>
body{font-family:-apple-system,sans-serif;background:#111;color:#eee;margin:0;padding:16px}
.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
img{max-width:48%;height:auto;border:1px solid #444;border-radius:6px}
.row{display:flex;gap:10px;flex-wrap:wrap}
.panel{background:#1c1c1c;padding:12px;border-radius:8px;flex:1;min-width:280px}
.d{display:inline-block;width:60px;font-weight:bold}
.ok{color:#ff6b6b}.no{color:#888}small{color:#999}
button{background:#2a2a2a;color:#eee;border:1px solid #444;padding:8px 16px;border-radius:6px;cursor:pointer}
button:hover{background:#3a3a3a}
.gtbar{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;align-items:center}
.gbtn{font-size:18px;padding:10px 18px;border-radius:8px;min-width:56px}
.gbtn.on{background:#c0392b;border-color:#e74c3c;color:#fff}.gbtn.off{background:#2a2a2a;color:#888}
.gsave{background:#27ae60;font-weight:bold;border-color:#2ecc71}.gsave:hover{background:#2ecc71}
.kbd{background:#333;padding:1px 5px;border-radius:3px;font-family:monospace;font-size:11px}
#tip{position:absolute;display:none;background:rgba(0,0,0,.85);color:#ffe28a;padding:3px 8px;border-radius:4px;font-size:13px;font-family:monospace;pointer-events:none;z-index:10}
#dimg{cursor:crosshair}
</style></head><body>
<div id="tip"></div>
<div class="bar">
  <button onclick="go(__IM1__)">◀</button><b>__IP1__ / __N__</b><button onclick="go(__IP1__)">▶</button>
  <button onclick="location='/?v=__VTOGGLE__'">🔁 __VTOGGLELABEL__</button>
</div>
<div class="row">
  <div><img src="/img?p=__U__"><br><small>原图</small></div>
  <div><img id="dimg" src="/depth?p=__U__"><br><small>深度图（白线=5区域边界，悬停看米数）</small></div>
  <div class="panel">
    <div style="word-break:break-all">__NAME__</div>
    <div>阈值：__THS__ | 占比&gt;__FR__ | 环境：<b>__ENV__</b></div>
    <div id="fracs">加载区域占比中…</div>
    <div class="gtbar">
      <button class="gbtn __C0__" onclick="tg(0)">前<span class="kbd">1</span></button>
      <button class="gbtn __C1__" onclick="tg(1)">左<span class="kbd">2</span></button>
      <button class="gbtn __C2__" onclick="tg(2)">右<span class="kbd">3</span></button>
      <button class="gbtn __C3__" onclick="tg(3)">上<span class="kbd">4</span></button>
      <button class="gbtn __C4__" onclick="tg(4)">下<span class="kbd">5</span></button>
      <button class="gsave" onclick="sv(false)">保存<span class="kbd">S</span></button>
      <button class="gsave" onclick="sv(true)">保存+下张<span class="kbd">↵</span></button>
    </div>
  </div>
</div>
<script>
fetch('/fracs?p=__U__').then(r=>r.json()).then(d=>{let s='';for(const k of ['前','左','右','上','下']){const v=d[k];const pct=(v.frac*100).toFixed(1);const hit=v.label?'<span class=ok>●障碍</span>':'<span class=no>○</span>';s+=`<div><span class=d>${k}</span>${hit} 近距占比 ${pct}% (阈 ${v.thresh.toFixed(1)}m)</div>`;}document.getElementById('fracs').innerHTML=s;});
(function(){const img=document.getElementById('dimg'),tip=document.getElementById('tip');let dd=null,W=0,H=0;fetch('/depthvals?p=__U__').then(r=>{W=+r.headers.get('X-W');H=+r.headers.get('X-H');return r.arrayBuffer();}).then(b=>{dd=new Float32Array(b);});img.addEventListener('mousemove',e=>{if(!dd)return;const r=img.getBoundingClientRect();const ix=Math.min(W-1,Math.max(0,Math.floor((e.clientX-r.left)/r.width*W)));const iy=Math.min(H-1,Math.max(0,Math.floor((e.clientY-r.top)/r.height*H)));const v=dd[iy*W+ix];tip.style.display='block';tip.style.left=(e.pageX+14)+'px';tip.style.top=(e.pageY+14)+'px';tip.textContent=Number.isFinite(v)?v.toFixed(2)+' m':'— m';});img.addEventListener('mouseleave',()=>{tip.style.display='none';});})();
const DIRS=['前','左','右','上','下'];let gt={'前':__B0__,'左':__B1__,'右':__B2__,'上':__B3__,'下':__B4__};
function tg(i){const d=DIRS[i];gt[d]=!gt[d];document.querySelectorAll('.gbtn')[i].className='gbtn '+(gt[d]?'on':'off');}
function sv(next){fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({relative_path:'__RP__','前':gt['前'],'左':gt['左'],'右':gt['右'],'上':gt['上'],'下':gt['下']})}).then(r=>r.json()).then(x=>{if(!x.ok){alert('保存失败: '+x.error);return;}location='/?v=__V__&i='+(next?__IP1__:__IP0__);});}
function go(i){if(i<0)i=0;if(i>__NM1__)i=__NM1__;location='/?v=__V__&i='+i;}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const m={'1':0,'2':1,'3':2,'4':3,'5':4};if(e.key in m)tg(m[e.key]);else if(e.key==='s'||e.key==='S')sv(false);else if(e.key==='Enter')sv(true);else if(e.key==='ArrowLeft')go(__IM1__);else if(e.key==='ArrowRight')sv(true);});
</script></body></html>"""


def render_html(i, n, key, v):
    rec = DATA.get(key, {})
    cur = {d: bool(rec.get(d)) for d in DIRS}
    env = _env_of(key)
    total = len(ALL_KEYS)
    other_v, other_label = ("all", f"看全部 {total}") if v == "obs" else ("obs", "只看有障碍")
    h = (HTML
         .replace("__IM1__", str(i-1)).replace("__IP1__", str(i+1)).replace("__IP0__", str(i))
         .replace("__NM1__", str(n-1)).replace("__N__", str(n))
         .replace("__U__", quote(key, safe="")).replace("__NAME__", key.split("/")[-1])
         .replace("__THS__", "  ".join(f"{d}&lt;{DIST_THRESH[d]}m" for d in DIRS))
         .replace("__ENV__", "室内" if env == "indoor" else "室外")
         .replace("__FR__", str(FRAC_THRESH))
         .replace("__V__", v).replace("__VTOGGLE__", other_v).replace("__VTOGGLELABEL__", other_label)
         .replace("__RP__", json.dumps(key)))
    for j, d in enumerate(DIRS):
        h = h.replace(f"__C{j}__", "on" if cur[d] else "off")
        h = h.replace(f"__B{j}__", "true" if cur[d] else "false")
    return h.encode("utf-8")


class Handler(SilentHandler):
    def do_HEAD(self):
        if urlparse(self.path).path == "/video":
            serve_video(self, BASE, urlparse(self.path).query, head_only=True)
        else:
            send(self, 200)

    def do_GET(self):
        parsed = urlparse(self.path); qs = parse_qs(parsed.query)
        try:
            if parsed.path == "/img":
                serve_image(self, BASE, parsed.query)
            elif parsed.path == "/depth":
                send(self, 200, render_depth_png(qs["p"][0]), "image/png")
            elif parsed.path == "/fracs":
                key = qs["p"][0]
                fracs = region_close_frac(get_depth_array(key))
                rec = DATA.get(key, {})
                out = {d: {"frac": fracs[d][0], "thresh": fracs[d][1],
                           "label": bool(rec.get(d))} for d in DIRS}
                send(self, 200, json.dumps(out).encode(), "application/json")
            elif parsed.path == "/depthvals":
                stride = 4
                small = get_depth_array(qs["p"][0])[::stride, ::stride]
                send(self, 200, small.astype(np.float32).tobytes(), "application/octet-stream",
                     headers={"X-W": str(small.shape[1]), "X-H": str(small.shape[0]),
                              "Access-Control-Allow-Origin": "*"})
            elif parsed.path in ("/", "/index.html"):
                v = qs.get("v", ["obs"])[0]
                keys = OBS_KEYS if v == "obs" else ALL_KEYS
                if not keys:
                    keys = ALL_KEYS
                n = len(keys)
                i = max(0, min(n-1, int(qs.get("i", ["0"])[0])))
                send(self, 200, render_html(i, n, keys[i], v), "text/html; charset=utf-8")
            else:
                send(self, 404)
        except Exception as e:
            import traceback; traceback.print_exc()
            send(self, 500, f"错误: {e}".encode())

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
    global _cfg, BASE, DEPTH_CACHE, JSON_PATH, SAVE_GT, DATA, ALL_KEYS, OBS_KEYS
    json_name = next((a for a in sys.argv[1:] if a.endswith(".json")), None)
    _cfg = load_config()
    port = next((int(a) for a in sys.argv[1:] if a.isdigit()), None) \
        or _cfg.viewer.default_port_depth
    BASE = Path(_cfg.paths.base)
    DEPTH_CACHE = Path(_cfg.paths.depth_maps_cache)
    if json_name:
        _cfg.outputs.depth_annotations = json_name   # point at chosen file
    JSON_PATH = BASE / (json_name or _cfg.outputs.depth_annotations)
    SAVE_GT = make_save_gt(JSON_PATH, "predicted", "predicted_model", _save_lock)
    DATA = json.load(open(JSON_PATH, encoding="utf-8"))
    ALL_KEYS = sorted(DATA)
    OBS_KEYS = [k for k in ALL_KEYS if any(DATA[k].get(d) for d in DIRS)]
    banner = (f"=" * 64 + f"\n  深度图浏览器 + GT 人工校验\n"
              f"  JSON: {JSON_PATH.name} | {len(ALL_KEYS)} 张 | 有障碍 {len(OBS_KEYS)}\n"
              f"  1-5 切方向 · S/Enter 保存 · 原值快照 predicted\n" + "=" * 64)
    run_server(port, Handler, JSON_PATH.name, banner)


if __name__ == "__main__":
    main()
