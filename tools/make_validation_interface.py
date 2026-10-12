#!/usr/bin/env python3
"""Generate the VLM obstacle-detection GT spot-check HTML.

Static (file://) mode writes /tmp/vlm_gt_validation_interface.html (offline).
Served mode (http://) is driven by tools/serve_validation.py — submissions
write back to the loaded JSON (snapshot key 'vlm_predicted').

Camera-label parsing (运镜类型 → 运镜方向, 场景类别) is unique to this tool.
"""
import glob
import html as _html
import json
import os
import random
import re
import sys
from datetime import datetime
from urllib.parse import quote

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from vlm_intention.common.config import load_config
from vlm_intention.common.paths import mp4_for, resolve_base
from vlm_intention.common.schema import DIRS

STATIC_OUT = "/tmp/vlm_gt_validation_interface.html"

# 运镜类型 -> 运镜方向 keyword rules (叠加，'+' 连接)
_DIR_RULES = [
    ("前（推进）", ["前推", "推进"]),
    ("后（拉出）", ["后拉", "拉出", "反冲"]),
    ("横向（左/右）", ["横移", "侧移", "侧跟", "侧摇", "横摇"]),
    ("竖向（上/下）", ["竖摇", "上摇", "下摇", "竖移", "升降", "升镜", "降镜"]),
    ("环绕", ["环绕", "环视", "盗梦", "格莱美", "旋转"]),
    ("快速摇", ["甩镜", "快速摇", "摇摄"]),
    ("静止", ["静止", "静态"]),
]
_CAMERA_CACHE = None


def parse_label_string(s):
    out = {}
    for part in re.split("[；;]", s or ""):
        m = re.match(r"\s*([^：:]+)[：:]\s*(.+)$", part.strip())
        if m:
            out[m.group(1).strip()] = m.group(2).strip()
    return out


def derive_direction(movement_type):
    if not movement_type:
        return ""
    hits = [label for label, kws in _DIR_RULES if any(kw in movement_type for kw in kws)]
    return " + ".join(hits) if hits else "未分类"


def load_camera_labels(base):
    global _CAMERA_CACHE
    if _CAMERA_CACHE is not None:
        return _CAMERA_CACHE
    out = {}
    for f in glob.glob(os.path.join(str(base), "*", "*", "标注tag", "video_labels.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict):
            for name, s in d.items():
                if isinstance(s, str):
                    out[name] = parse_label_string(s)
    _CAMERA_CACHE = out
    return out


def camera_info_for(base, key):
    raw = load_camera_labels(base).get(os.path.basename(mp4_for(key)), {})
    mtype = raw.get("运镜类型", "")
    return {"运镜类型": mtype, "运镜方向": derive_direction(mtype),
            "场景类别": raw.get("场景类别", "")}


def vlm_of(entry):
    src = entry.get("vlm_predicted")
    if isinstance(src, dict):
        return [d for d in DIRS if src.get(d)]
    return [d for d in DIRS if entry.get(d)]


def load_data(json_path):
    return json.load(open(json_path, encoding="utf-8"))


def pick_sample(data, base, n, seed, exclude=None):
    exclude = exclude or set()
    keys = [k for k in data if data[k].get("image_name", "") not in exclude
            and os.path.exists(os.path.join(str(base), k))]
    return random.Random(seed).sample(keys, min(n, len(keys)))


def build_validation_data(sample, data):
    vd = {}
    for i, k in enumerate(sample, start=1):
        v = data[k]
        vlm = vlm_of(v)
        corrected = bool(v.get("corrected"))
        gt = [d for d in DIRS if v.get(d)] if corrected else []
        vd[str(i)] = {
            "image_name": v.get("image_name", ""), "relative_path": v.get("relative_path", k),
            "model": v.get("vlm_model") or v.get("model", ""), "vlm_detection": vlm,
            "gt_detection": gt, "touched": corrected, "completed": corrected,
            "match": (sorted(vlm) == sorted(gt)) if corrected else None,
        }
    return vd


def sample_block(base, i, k, v, vlm, gt, served=False, cam=None):
    name = _html.escape(v.get("image_name", ""))
    src = _html.escape("/img?p=" + quote(k, safe="")) if served \
        else _html.escape("file://" + os.path.join(str(base), k))
    scene = _html.escape(k.split("/")[1] if len(k.split("/")) > 1 else "")
    mp4_rel = mp4_for(k)
    mp4_exists = os.path.isfile(os.path.join(str(base), mp4_rel))
    if mp4_exists:
        vsrc = _html.escape("/video?p=" + quote(mp4_rel, safe="")) if served \
            else _html.escape("file://" + os.path.join(str(base), mp4_rel))
        video_tag = (f'<video class="media-display" controls preload="metadata" playsinline poster="{src}">'
                     f'<source src="{vsrc}" type="video/mp4">浏览器不支持 video。</video>')
    else:
        video_tag = '<div class="no-video">⚠️ 未找到对应 mp4</div>'
    vlm_tags = "".join(f'<span class="obstacle-tag obstacle-{"yes" if d in vlm else "no"}">{d}{"✓" if d in vlm else "✗"}</span>' for d in DIRS)
    gt_tags = "".join(f'<span class="gt-obstacle-tag {"gt-obstacle-yes" if d in gt else "gt-obstacle-no"}" id="gt-{i}-{d}" onclick="toggleGT({i}, \'{d}\')">{d}{"✓" if d in gt else "?"}</span>' for d in DIRS)
    cam = cam or {}
    return f"""
        <div class="sample-container" id="sample-{i}">
            <div class="sample-header">【样本 {i}】{name}
                <span class="scene-badge">📁 {scene}</span>
                <span class="vlm-badge">🤖 VLM: {_html.escape(", ".join(vlm) if vlm else "（无）")}</span>
            </div>
            <div class="camera-info">
                <span><span class="ci-label">🎬 运镜类型：</span><b>{_html.escape(cam.get("运镜类型","") or "—")}</b></span>
                <span><span class="ci-label">🧭 运镜方向：</span><b>{_html.escape(cam.get("运镜方向","") or "—")}</b></span>
                <span><span class="ci-label">🏷 场景类别：</span><b>{_html.escape(cam.get("场景类别","") or "—")}</b></span>
            </div>
            <div class="media-row">
                <div class="media-cell"><div class="media-label">🖼 抽取帧</div>
                    <img src="{src}" class="media-display" alt="样本"></div>
                <div class="media-cell"><div class="media-label">🎬 原视频（可播放）</div>{video_tag}</div>
            </div>
            <div class="comparison">
                <div class="detection-box vlm-result"><strong>🔴 VLM 检测（参考）</strong><div class="obstacle-tags">{vlm_tags}</div></div>
                <div class="detection-box gt-result"><strong>🔵 Ground Truth（点击 → 提交写回 JSON）</strong><div class="obstacle-tags">{gt_tags}</div></div>
            </div>
            <div class="quick-actions">
                <button class="quick-btn no-obstacle-btn" onclick="setNoObstacles({i})">🚀 无障碍物</button>
                <button class="quick-btn reset-btn" onclick="resetGT({i})">🔄 重置</button>
            </div>
            <div class="match-indicator neutral" id="match-{i}">点击蓝色标签设置真实障碍物状态</div>
            <button class="submit-btn" id="submit-{i}" onclick="submitGT({i})">提交 GT（写回 JSON）</button>
        </div>"""


CSS = """
body{font-family:Arial,"PingFang SC",sans-serif;margin:20px;background:#f5f5f5}
.header{background:#2c3e50;color:white;padding:20px;border-radius:8px;margin-bottom:12px}
.info{background:#ecf0f1;color:#2c3e50;padding:10px 15px;border-radius:6px;margin-bottom:10px;font-size:13px}
#flash{min-height:22px;font-weight:bold}#flash.ok{color:#27ae60}#flash.err{color:#c0392b}
.stats{display:flex;flex-wrap:wrap;justify-content:space-around;background:white;padding:15px;border-radius:8px;margin-bottom:16px;gap:8px}
.stat-box{text-align:center;min-width:110px}.stat-number{font-size:24px;font-weight:bold;color:#3498db}
.sample-container{background:white;padding:20px;margin-bottom:15px;border-radius:8px;border-left:4px solid #3498db}
.sample-header{font-size:16px;font-weight:bold;margin-bottom:10px;color:#2c3e50}
.scene-badge,.vlm-badge{display:inline-block;font-size:12px;padding:2px 8px;border-radius:10px;margin-left:6px}
.scene-badge{background:#eef2f7;color:#5b6770}.vlm-badge{background:#fdecea;color:#c0392b}
.media-row{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin:15px 0;align-items:start}
@media(max-width:760px){.media-row{grid-template-columns:1fr}}
.media-label{font-size:12px;color:#5b6770;font-weight:bold;margin-bottom:4px}
.media-display{max-width:100%;max-height:360px;border:2px solid #ddd;border-radius:4px;display:block;background:#000}
.no-video{padding:40px 12px;text-align:center;color:#999;border:2px dashed #ddd;border-radius:4px}
.camera-info{background:#fff8e1;border:1px solid #ffe082;border-radius:6px;padding:8px 12px;margin:10px 0;font-size:13px;color:#5d4e1a;display:flex;flex-wrap:wrap;gap:6px 18px}
.camera-info b{color:#e65100}.camera-info .ci-label{color:#8d6e00}
.comparison{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin:15px 0}
.detection-box{padding:10px;border-radius:4px}.vlm-result{background:#e8f4f8}.gt-result{background:#f0f8f0}
.obstacle-tags{margin:10px 0}
.obstacle-tag,.gt-obstacle-tag{display:inline-block;padding:8px 12px;margin:3px;border-radius:5px;font-weight:bold;border:2px solid #ddd}
.obstacle-yes{background:#27ae60;color:white;border-color:#27ae60}.obstacle-no{background:#ecf0f1;color:#7f8c8d}
.gt-obstacle-tag{cursor:pointer;border:2px solid #3498db;transition:all .2s}.gt-obstacle-tag:hover{transform:scale(1.05)}
.gt-obstacle-yes{background:#3498db;color:white;border-color:#3498db}.gt-obstacle-no{background:#ecf0f1;color:#7f8c8d}
.match-indicator{text-align:center;padding:10px;margin:10px 0;border-radius:4px;font-weight:bold}
.match-yes{background:#d4edda;color:#155724}.match-no{background:#f8d7da;color:#721c24}.neutral{background:#f8f9fa;color:#555;font-weight:normal}
.quick-actions{text-align:center;margin:15px 0;padding:10px;background:#f8f9fa;border-radius:4px}
.quick-btn{padding:8px 15px;margin:5px;border:none;border-radius:4px;cursor:pointer;font-weight:bold;color:white}
.no-obstacle-btn{background:#95a5a6}.reset-btn{background:#e74c3c}
.submit-btn{width:100%;padding:12px;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:16px;background:#3498db;color:white}
.submit-btn:hover{background:#2980b9}.submit-btn.completed{background:#27ae60}.submit-btn.saved{background:#16a085}
.progress-bar{width:100%;height:20px;background:#ecf0f1;border-radius:10px;overflow:hidden;margin:10px 0}
.progress-fill{height:100%;background:linear-gradient(90deg,#3498db,#2ecc71);transition:width .3s}
"""

JS = r"""
const validationData=__VD__;const DIRS=["前","左","右","上","下"];let savedCount=0;const SERVED=window.location.protocol.startsWith('http');
function flash(m,k){const e=document.getElementById('flash');e.textContent=m||'';e.className='info '+(k||'');if(m)clearTimeout(window.__t),window.__t=setTimeout(()=>{e.textContent='';e.className='info';},4000);}
function toggleGT(n,d){const s=validationData[n];s.touched=true;const tag=document.getElementById(`gt-${n}-${d}`);const a=s.gt_detection;if(tag.classList.contains('gt-obstacle-no')){tag.classList.replace('gt-obstacle-no','gt-obstacle-yes');tag.textContent=d+'✓';if(!a.includes(d))a.push(d);}else{tag.classList.replace('gt-obstacle-yes','gt-obstacle-no');tag.textContent=d+'?';const i=a.indexOf(d);if(i>-1)a.splice(i,1);}updateMatch(n);}
function setNoObstacles(n){const s=validationData[n];s.touched=true;DIRS.forEach(d=>{const t=document.getElementById(`gt-${n}-${d}`);t.className='gt-obstacle-tag gt-obstacle-no';t.textContent=d+'?';});s.gt_detection=[];updateMatch(n);}
function resetGT(n){const s=validationData[n];DIRS.forEach(d=>{const t=document.getElementById(`gt-${n}-${d}`);t.className='gt-obstacle-tag gt-obstacle-no';t.textContent=d+'?';});s.gt_detection=[];s.touched=false;updateMatch(n);}
function updateMatch(n){const s=validationData[n];const v=[...s.vlm_detection].sort(),g=[...s.gt_detection].sort();const div=document.getElementById(`match-${n}`);if(!s.touched){div.textContent='点击蓝色标签设置真实障碍物状态';div.className='match-indicator neutral';return;}const fa=v.filter(x=>!g.includes(x)),mi=g.filter(x=>!v.includes(x));if(g.length===0){if(v.length===0){div.textContent='✅ 确认无障碍物（真阴性）';div.className='match-indicator match-yes';}else{div.textContent='❌ VLM 误报: '+fa.join(', ');div.className='match-indicator match-no';}}else if(fa.length===0&&mi.length===0){div.textContent='✅ VLM 与 GT 完全一致';div.className='match-indicator match-yes';}else{const iss=[];if(mi.length)iss.push('漏检: '+mi.join(', '));if(fa.length)iss.push('误报: '+fa.join(', '));div.textContent='❌ 差异: '+iss.join(' | ');div.className='match-indicator match-no';}}
async function submitGT(n){const s=validationData[n];if(!s.touched){flash('请先标注再提交','err');return;}const flags={};DIRS.forEach(d=>flags[d]=s.gt_detection.includes(d));const btn=document.getElementById(`submit-${n}`);let saved=false;if(SERVED){btn.disabled=true;btn.textContent='⏳ 写回...';try{const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(Object.assign({relative_path:s.relative_path},flags))});const j=await r.json();if(!j.ok){btn.disabled=false;btn.textContent='提交 GT（写回 JSON）';flash('写回失败: '+(j.error||r.status),'err');return;}saved=true;}catch(e){btn.disabled=false;btn.textContent='提交 GT（写回 JSON）';flash('写回出错: '+e,'err');return;}}s.completed=true;s.match=JSON.stringify([...s.vlm_detection].sort())===JSON.stringify([...s.gt_detection].sort());btn.textContent=saved?'✅ 已写回':'✅ 已提交（离线）';btn.classList.add(saved?'saved':'completed');btn.disabled=true;if(saved){savedCount++;document.getElementById('save-count').textContent=savedCount;flash('已写回：'+s.image_name,'ok');}updateStats();}
function updateStats(){const v=Object.values(validationData);const c=v.filter(x=>x.completed).length;const ok=v.filter(x=>x.completed&&x.match).length;document.getElementById('completed-count').textContent=c;document.getElementById('accuracy-rate').textContent=(c?Math.round(ok/c*100):0)+'%';document.getElementById('pending-count').textContent=v.filter(x=>!x.completed).length;document.getElementById('progress-fill').style.width=(c/v.length*100)+'%';}
document.addEventListener('DOMContentLoaded',()=>{Object.keys(validationData).forEach(n=>{updateMatch(parseInt(n));});if(!SERVED)flash('⚠️ 离线 file:// 模式，提交不写回 JSON。用 serve_validation.py 启服务。','err');});
"""


def build_html(base, sample, data, vd, seed, total, served=False):
    blocks = [sample_block(base, i, k, data[k], vd[str(i)]["vlm_detection"],
                           vd[str(i)]["gt_detection"], served, camera_info_for(base, k))
              for i, k in enumerate(sample, start=1)]
    n = len(sample)
    has_obs = sum(1 for k in sample if any(data[k].get(d) for d in DIRS))
    info = (f"数据集总数 {total} · 抽样 {n} · seed={seed} · "
            f"含障碍 {has_obs} / 无障碍 {n-has_obs} · {datetime.now():%Y-%m-%d %H:%M}")
    served_hint = ("🟢 服务模式：提交直接写回 JSON（原值快照 vlm_predicted）。" if served
                   else "⚪ 离线模式：提交不写回。用 serve_validation.py 启服务。")
    shell = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0"><title>VLM障碍物 - GT 抽查</title>
<style>__CSS__</style></head><body>
<div class="header"><h1>🎯 VLM 障碍物检测 — Ground Truth 抽查</h1>
<p>点击蓝色标签设置真实障碍物，提交即写回 JSON。</p></div>
<div class="info">__INFO__</div><div class="info" id="flash">__SERVED__</div>
<div class="stats">
<div class="stat-box"><div class="stat-number">__N__</div><div>总样本</div></div>
<div class="stat-box"><div class="stat-number" id="completed-count">0</div><div>✅ 已标注</div></div>
<div class="stat-box"><div class="stat-number" id="accuracy-rate">0%</div><div>📊 精确匹配</div></div>
<div class="stat-box"><div class="stat-number" id="save-count">0</div><div>💾 已写回</div></div>
<div class="stat-box"><div class="stat-number" id="pending-count">__N__</div><div>⏳ 待标注</div></div>
</div>
<div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
<div id="samples-container">__BLOCKS__</div>
<script>__JS__</script></body></html>"""
    return (shell.replace("__CSS__", CSS)
            .replace("__JS__", JS.replace("__VD__", json.dumps(vd, ensure_ascii=False)))
            .replace("__INFO__", _html.escape(info)).replace("__SERVED__", served_hint)
            .replace("__BLOCKS__", "".join(blocks)).replace("__N__", str(n)))


def main():
    cfg = load_config()
    base = resolve_base(cfg)
    json_path = base / (os.environ.get("OBSTACLE_JSON") or cfg.outputs.vlm_annotations)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    data = load_data(json_path)
    sample = pick_sample(data, base, n, seed)
    vd = build_validation_data(sample, data)
    html_doc = build_html(base, sample, data, vd, seed, len(data))
    with open(STATIC_OUT, "w", encoding="utf-8") as f:
        f.write(html_doc)
    has_obs = sum(1 for k in sample if any(data[k].get(d) for d in DIRS))
    print(f"✅ 静态界面: {STATIC_OUT} | 样本 {len(sample)} | 含障碍 {has_obs} | seed={seed}")
    print(f"   打开: file://{STATIC_OUT}")


if __name__ == "__main__":
    main()
