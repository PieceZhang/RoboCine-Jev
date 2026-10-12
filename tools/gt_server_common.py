#!/usr/bin/env python3
"""Shared HTTP spine for the GT/viewer servers.

Removes the ~250 lines of duplication across serve_depth_viewer /
serve_validation / serve_mined_gt: a silent BaseHTTPRequestHandler base, the
_send helper, HTTP-Range video streaming (206/200/416), and run_server().
"""
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from vlm_intention.common.paths import safe_resolve


class SilentHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass


class GtHandler(SilentHandler):
    """Base for the GT servers: handles HEAD for /video (Range probe).
    Subclasses set class attr BASE."""
    BASE = None

    def do_HEAD(self):
        if urlparse(self.path).path == "/video":
            serve_video(self, self.BASE, urlparse(self.path).query, head_only=True)
        else:
            send(self, 200)


# ── shared 5-direction GT-toggle widget (used by serve_depth_viewer + serve_mined_gt) ──

def gt_toggle_css():
    """CSS for the .gbtn/.gsave/.kbd toggle widget."""
    return (".gtbar{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;align-items:center}"
            ".gbtn{font-size:18px;padding:10px 18px;border-radius:8px;min-width:56px}"
            ".gbtn.on{background:#c0392b;border-color:#e74c3c;color:#fff}"
            ".gbtn.off{background:#2a2a2a;color:#888}"
            ".gsave{background:#27ae60;font-weight:bold;border-color:#2ecc71}"
            ".gsave:hover{background:#2ecc71}"
            ".kbd{background:#333;padding:1px 5px;border-radius:3px;font-family:monospace;font-size:11px}")


def gt_toggle_buttons():
    """The 7-button row (5 dirs + 保存 + 保存+下张) with __C0__..__C4__ placeholders.
    Wrap in the server's own container div."""
    return ('<button class="gbtn __C0__" onclick="tg(0)">前<span class="kbd">1</span></button>'
            '<button class="gbtn __C1__" onclick="tg(1)">左<span class="kbd">2</span></button>'
            '<button class="gbtn __C2__" onclick="tg(2)">右<span class="kbd">3</span></button>'
            '<button class="gbtn __C3__" onclick="tg(3)">上<span class="kbd">4</span></button>'
            '<button class="gbtn __C4__" onclick="tg(4)">下<span class="kbd">5</span></button>'
            '<button class="gsave" onclick="sv(false)">保存<span class="kbd">S</span></button>'
            '<button class="gsave" onclick="sv(true)">保存+下张<span class="kbd">↵</span></button>')


def gt_toggle_js():
    """The tg/sv/go/keydown JS body (NO <script> wrapper — embed in a script tag).
    Placeholders: __B0__..__B4__ (true/false), __RP__ (json-quoted key),
    __V__/__IM1__/__IP0__/__IP1__/__NM1__."""
    return ("const DIRS=['前','左','右','上','下'];"
            "let gt={'前':__B0__,'左':__B1__,'右':__B2__,'上':__B3__,'下':__B4__};"
            "function tg(i){const d=DIRS[i];gt[d]=!gt[d];document.querySelectorAll('.gbtn')[i].className='gbtn '+(gt[d]?'on':'off');}"
            "function sv(next){fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},"
            "body:JSON.stringify({relative_path:__RP__,'前':gt['前'],'左':gt['左'],'右':gt['右'],'上':gt['上'],'下':gt['下']})})"
            ".then(r=>r.json()).then(x=>{if(!x.ok){alert('保存失败: '+x.error);return;}location='/?v=__V__&i='+(next?__IP1__:__IP0__);});}"
            "function go(i){if(i<0)i=0;if(i>__NM1__)i=__NM1__;location='/?v=__V__&i='+i;}"
            "document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;"
            "const m={'1':0,'2':1,'3':2,'4':3,'5':4};if(e.key in m)tg(m[e.key]);"
            "else if(e.key==='s'||e.key==='S')sv(false);else if(e.key==='Enter')sv(true);"
            "else if(e.key==='ArrowLeft')go(__IM1__);else if(e.key==='ArrowRight')sv(true);});")


def send(handler, code, body=b"", ctype="text/plain; charset=utf-8", headers=None):
    handler.send_response(code)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    for k, v in (headers or {}).items():
        handler.send_header(k, v)
    handler.end_headers()
    if body:
        handler.wfile.write(body)


def serve_image(handler, base, query):
    rel = (parse_qs(query).get("p") or [""])[0]
    full, ok = safe_resolve(base, rel)
    if not ok:
        send(handler, 404); return
    ct = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
          ".webp": "image/webp"}.get(full[full.rfind("."):].lower(), "application/octet-stream")
    with open(full, "rb") as f:
        send(handler, 200, f.read(), ct)


def serve_video(handler, base, query, head_only=False):
    """HTTP Range mp4 streaming: 206 partial / 200 full / 416 unsatisfiable."""
    rel = (parse_qs(query).get("p") or [""])[0]
    full, ok = safe_resolve(base, rel)
    if not rel:
        send(handler, 400); return
    if full is None:
        send(handler, 403); return
    if not ok:
        send(handler, 404); return
    size = __import__("os").path.getsize(full)
    start, end = 0, size - 1
    partial = False
    rng = handler.headers.get("Range")
    if rng and rng.startswith("bytes="):
        partial = True
        spec = rng[6:].split("-", 1)
        try:
            start = int(spec[0]) if spec[0] else 0
            end = int(spec[1]) if (len(spec) > 1 and spec[1]) else size - 1
        except ValueError:
            partial = False
        if partial and (start < 0 or start >= size or end >= size or start > end):
            handler.send_response(416)
            handler.send_header("Content-Range", f"bytes */{size}")
            handler.end_headers()
            return
    length = end - start + 1
    handler.send_response(206 if partial else 200)
    handler.send_header("Content-Type", "video/mp4")
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(length))
    if partial:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.end_headers()
    if head_only:
        return
    with open(full, "rb") as f:
        f.seek(start)
        rem = length
        while rem > 0:
            chunk = f.read(min(1 << 20, rem))
            if not chunk:
                break
            handler.wfile.write(chunk)
            rem -= len(chunk)


def run_server(port, Handler, json_name=None, banner=None, open_browser=True):
    if banner:
        print(banner)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"✅ 服务已启动 → {url}  (Ctrl+C 停止)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 已停止。")
        srv.shutdown()
