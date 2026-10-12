"""Offline 3D trajectory viewer: generates ONE self-contained HTML file
(vanilla-JS canvas renderer — no CDN, no server, opens over file://).

Draws the source point cloud (downsampled, height-colored, embedded as
quantized base64), rooms / OBB objects with labeled centers, anchor
selections with scores, the subject track + animated capsule, the camera
trajectory colored per segment with ghost view-frustums along the path
(per-frame FOV, 16:9), collision & occlusion markers, and a timeline
slider that flies the camera frustum along the path. Keys: P toggles the
cloud, L toggles labels/markers.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vlm_intention.common.atomic_io import load_json
from vlm_intention.common.config import load_config

CLOUD_MAX_POINTS = 60000            # embedded-cloud budget (HTML size / fps);
CLOUD_VOXEL = 0.05                  # m, first-pass downsample before the cap.
                                    # both overridable per run:
                                    # --cloud-voxel / --cloud-max-points

# in-viewer cloud density modes (微/少/密/无 buttons, B key):
#   微 = 1/16 of base embed · 少 = 1/4 · 密 = FULL cloud (uncapped 2nd embed)
#   无 = cloud hidden (boxes/trajectory only)
CLOUD_MODE_STEPS = (16, 4, 1, 0)


def pack_point_cloud(points: np.ndarray, max_pts: int = CLOUD_MAX_POINTS,
                     voxel: float = CLOUD_VOXEL,
                     colors: np.ndarray | None = None) -> dict | None:
    """Downsample + quantize a cloud for HTML embedding.

    Int16 millimeter-ish quantization against a per-cloud origin/scale keeps
    ~30k points at ~180 KB base64. Optional per-point RGB u8 colors
    (colors[i] ↔ points[i]) ride through the same keep/subsample as positions
    (as key ``c64``), enabling the viewer's true-color clean mode.
    Returns {origin, scale, count, b64[, c64]} or None when nothing to embed.
    """
    if points is None or len(points) < 4:
        return None
    pts = np.asarray(points, dtype=np.float64)
    rgb = None if colors is None else np.asarray(colors, dtype=np.uint8)
    # same selection as extract.voxel_downsample, but carrying colors along
    keys = np.floor(pts / voxel).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(keep)
    pts = pts[keep]
    if rgb is not None:
        rgb = rgb[keep]
    if max_pts and len(pts) > max_pts:                  # uniform subsample
        rng = np.random.default_rng(0)
        sel = rng.choice(len(pts), max_pts, replace=False)
        pts = pts[sel]
        if rgb is not None:
            rgb = rgb[sel]
    origin = pts.min(axis=0)
    extent = float((pts.max(axis=0) - origin).max())
    scale = max(extent / 32767.0, 1e-6)                 # fit Int16 range
    q = np.round((pts - origin) / scale).astype("<i2")
    out = {"origin": [float(v) for v in origin], "scale": float(scale),
           "count": int(len(pts)),
           "b64": base64.b64encode(q.tobytes()).decode("ascii")}
    if rgb is not None:
        out["c64"] = base64.b64encode(rgb.tobytes()).decode("ascii")
    return out


# 3DGS PLY vertex layout: 14 float32 (x,y,z, f_dc_0..2, opacity, scale_0..2, rot_0..3)
_SH_C0 = 0.2820547913868781          # degree-0 spherical harmonics constant


def _read_gs_colors(path: Path) -> np.ndarray | None:
    """Per-vertex sRGB u8 colors from a 3DGS PLY (f_dc → r = 0.5 + SH_C0·f_dc).
    Returns None (silently) when the layout isn't a plain-float 3DGS dump.
    Row order matches load_points exactly (no permutation)."""
    try:
        if path.suffix.lower() != ".ply":
            return None
        with open(path, "rb") as fh:
            header = b""
            while not header.endswith((b"end_header\n", b"end_header\r\n")):
                byte = fh.read(1)
                if not byte:                 # EOF before the marker — not a
                    return None              # PLY (npz/pcd/…) → no color layer
                header += byte
            h = header.decode("latin1")
            props = [l.split()[-1] for l in h.splitlines()
                     if l.startswith("property float")]
            n = int([l for l in h.splitlines()
                     if l.startswith("element vertex")][0].split()[-1])
            need = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
            if not all(p in props for p in need):
                return None
            dt = np.dtype([(p, "<f4") for p in props])
            verts = np.frombuffer(fh.read(n * dt.itemsize), dtype=dt, count=n)
            rgb = np.stack([np.clip(0.5 + _SH_C0 * verts[f]
                                    .astype(np.float64), 0.0, 1.0)
                            for f in ("f_dc_0", "f_dc_1", "f_dc_2")], axis=1)
            return np.round(rgb * 255).astype(np.uint8)
    except Exception:                                       # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# HTML template (vanilla canvas 3D — painter's algorithm)
# --------------------------------------------------------------------------- #
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>cinema_traj · __SCENE_ID__</title>
<style>
  html,body{margin:0;height:100%;background:#0e1116;color:#cdd3dd;
            font:13px/1.45 -apple-system,"PingFang SC",sans-serif;overflow:hidden}
  #c{display:block;width:100%;height:100%;cursor:grab}
  #glcv{width:100%;height:100%}
  #hud{position:fixed;left:12px;top:12px;background:rgba(20,25,33,.88);
       border:1px solid #2a3342;border-radius:10px;padding:10px 14px;max-width:340px}
  #hud h1{font-size:14px;margin:0 0 6px;color:#8ab4ff}
  .m{display:grid;grid-template-columns:auto auto;gap:2px 12px;font-size:12px}
  .m b{color:#e8eaf0;font-weight:600}
  #legend{position:fixed;right:12px;top:12px;background:rgba(20,25,33,.88);
          border:1px solid #2a3342;border-radius:10px;padding:8px 12px;font-size:12px}
  .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
  #map{position:fixed;right:12px;bottom:56px;background:rgba(20,25,33,.88);
       border:1px solid #2a3342;border-radius:10px;padding:6px 8px 4px}
  #map .tt{font-size:10px;color:#7d8695;margin:0 0 3px 2px}
  .mroom{fill:rgba(90,120,180,.14);stroke:rgba(130,160,220,.5);stroke-width:1}
  .mroom.cur{fill:rgba(91,140,255,.28);stroke:#8ab4ff;stroke-width:1.6}
  .mlab{font:9px -apple-system,"PingFang SC",sans-serif;fill:#9fb2cc;pointer-events:none}
  .mdoor{fill:none;stroke:rgba(90,230,150,.65);stroke-width:1.2}
  .mdot{fill:rgba(90,230,150,.9)}
  .mtraj{fill:none;stroke:#ff3232;stroke-width:1.8;opacity:.75}
  .marc{fill:none;stroke-width:1;opacity:.45;stroke-dasharray:2 2}
  .msubj{fill:none;stroke:#ffbe50;stroke-width:1;opacity:.6;stroke-dasharray:3 2}
  .mcam{fill:#fff}
  .mcamlook{stroke:#fff;stroke-width:1.2;opacity:.8}
  #bar{position:fixed;left:0;right:0;bottom:0;background:rgba(16,20,27,.92);
       border-top:1px solid #2a3342;padding:8px 16px;display:flex;gap:12px;align-items:center}
  input[type=range]{flex:1;accent-color:#5b8cff}
  #tlabel{font-variant-numeric:tabular-nums;min-width:96px;text-align:right}
  button{background:#1b2330;color:#cdd3dd;border:1px solid #2a3342;
         border-radius:8px;padding:4px 14px;cursor:pointer}
</style>
</head>
<body>
<canvas id="glcv" style="position:absolute;inset:0"></canvas>
<canvas id="c" style="position:absolute;inset:0;background:transparent"></canvas>
<div id="hud">
  <h1>cinema_traj — __SCENE_ID__</h1>
  <div class="m" id="metrics"></div>
  <div style="margin-top:6px;font-size:11px;color:#7d8695">
     拖动旋转 · 滚轮缩放 · 空格播放 · P 点云开关 · L 标签 · B 云密度(微/少/密/无) · C 纯景 · 右下:房间邻接小地图</div>
</div>
<div id="legend"></div>
<div id="map"><div class="tt">俯视 · 房间邻接 (+X→ / +Z↓)</div>
  <svg id="mmap" width="224" height="150"></svg></div>
<div id="bar">
  <button id="play">▶</button>
  <input id="time" type="range" min="0" max="1000" value="0">
  <span id="tlabel">t=0.0s</span>
  <span style="font-size:11px;color:#7d8695;margin-left:10px">点云</span>
  <button class="bx" data-m="0">微</button>
  <button class="bx" data-m="1">少</button>
  <button class="bx" data-m="2">密</button>
  <button class="bx" data-m="3">无</button>
  <span style="font-size:11px;color:#7d8695;margin-left:16px">视图</span>
  <button id="clean">纯景</button>
</div>
<script>
const DATA = __DATA__;
const CLOUD_STEPS = __CLOUD_STEPS__;
const SEGCOLORS = ["#5b8cff","#39c48f","#f0a742","#e46fb1","#8f7bff","#4ecbe6",
                   "#d5e05b","#ff8b6b","#6be0a8","#c98bff","#7fd1ff","#ffb2e0"];
// ---------- 3D helpers ----------
const sub3=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const add3=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const mul3=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
const dot3=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm3=a=>{const n=Math.hypot(a[0],a[1],a[2])||1;return[a[0]/n,a[1]/n,a[2]/n];};
const addY=(p,y)=>[p[0],p[1]+y,p[2]];
const hexA=(hex,a)=>{const n=parseInt(hex.slice(1),16);
  return `rgba(${n>>16&255},${n>>8&255},${n&255},${a})`;};
function obbCorners(o){                       // o: {center, rotation, extents}
  // canonical convention (common/obb.py): R's COLUMNS are local axes in
  // world → world = center + R @ local. Rows of R multiply local components.
  const R=o.rotation,h=o.extents.map(x=>x/2),c=o.center,P=[];
  for(let sx=-1;sx<=1;sx+=2)for(let sy=-1;sy<=1;sy+=2)for(let sz=-1;sz<=1;sz+=2){
    const l=[sx*h[0],sy*h[1],sz*h[2]];
    P.push([c[0]+R[0][0]*l[0]+R[0][1]*l[1]+R[0][2]*l[2],
            c[1]+R[1][0]*l[0]+R[1][1]*l[1]+R[1][2]*l[2],
            c[2]+R[2][0]*l[0]+R[2][1]*l[1]+R[2][2]*l[2]]);
  }
  return P;                                   // order: x-,y-,z- nested
}
const FACES=[[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]];
let cam={yaw:.8,pitch:.55,dist:12,target:[0,1,0]};
function viewMatrix(){
  const cy=Math.cos(cam.yaw),sy=Math.sin(cam.yaw),cp=Math.cos(cam.pitch),sp=Math.sin(cam.pitch);
  const ex=cam.target[0]+cam.dist*cp*cy, ey=cam.target[1]+cam.dist*sp, ez=cam.target[2]+cam.dist*cp*sy;
  const f=norm3(sub3(cam.target,[ex,ey,ez])), r=norm3(cross3(f,[0,1,0])), u=cross3(r,f);
  return {ex,ey,ez,r,u,f};
}
function project(p,V,W,H){
  const d=sub3(p,[V.ex,V.ey,V.ez]);
  const x=dot3(d,V.r),y=dot3(d,V.u),z=dot3(d,V.f);
  if(z<0.05)return null;
  const f=W*0.9;
  return {x:W/2+f*x/z,y:H/2-f*y/z,z};
}
// ---------- scene model ----------
const rooms=DATA.scene.rooms||[],
      objs=(DATA.scene.objects||[]).map(o=>({o,c:obbCorners(o.obb)}));
const structs=(DATA.scene.structures||[]).map(o=>({o,c:obbCorners(o.obb)}));
const doors=(DATA.scene.doors||[]).map(o=>({o,c:obbCorners(o.obb)}));
const frames=DATA.frames||[];
const segNames=[...new Set(frames.map(f=>f.segment))];
const segDisp=n=>n.startsWith('arc')?('过渡 '+n.slice(4)):n;   // arcs are 1-3-1 transitions, not shots
const segColor={}; segNames.forEach((n,i)=>segColor[n]=SEGCOLORS[i%SEGCOLORS.length]);
const tMin=frames.length?frames[0].t:0, tMax=frames.length?frames[frames.length-1].t:tMin+1;
// subject polyline
const subj=(DATA.scene.subject_track&&DATA.scene.subject_track.frames)||[];
// ---------- top-down minimap: room boxes + door adjacency + live camera ----------
// Orientation matches the default 3D camera (which sits at the (+X,+Z) corner
// looking at the center): world +X → map right, +Z → map down, so the corner
// nearest the viewer is bottom-right and the far side is up — same as the
// ground plane reads on screen at first load.
const mmapSvg=document.getElementById('mmap'),mmapPanel=document.getElementById('map');
const NSMAP='http://www.w3.org/2000/svg';
let mapPx=null,mapPz=null,mapCamG=null,mapCurRoom=null;
const rcById={};
(function buildMap(){
  if(!rooms.length){mmapPanel.style.display='none';return;}
  const MW=224,MH=150,PAD=10;
  let x0=1e9,z0=1e9,x1=-1e9,z1=-1e9;
  rooms.forEach(r=>{x0=Math.min(x0,r.bounds_min[0]);x1=Math.max(x1,r.bounds_max[0]);
                    z0=Math.min(z0,r.bounds_min[2]);z1=Math.max(z1,r.bounds_max[2]);});
  const s=Math.min((MW-2*PAD)/Math.max(x1-x0,1e-6),(MH-2*PAD)/Math.max(z1-z0,1e-6));
  const ox=(MW-(x1-x0)*s)/2,oz=(MH-(z1-z0)*s)/2;   // center the room bbox
  mapPx=x=>ox+(x-x0)*s; mapPz=z=>oz+(z-z0)*s;
  const el=(t,at,parent)=>{const e=document.createElementNS(NSMAP,t);
    for(const k in at)e.setAttribute(k,at[k]);(parent||mmapSvg).appendChild(e);return e;};
  rooms.forEach(r=>{
    rcById[r.id]=el('rect',{x:mapPx(r.bounds_min[0]).toFixed(1),
      y:mapPz(r.bounds_min[2]).toFixed(1),
      width:Math.max(2,(r.bounds_max[0]-r.bounds_min[0])*s).toFixed(1),
      height:Math.max(2,(r.bounds_max[2]-r.bounds_min[2])*s).toFixed(1),rx:2,
      'class':'mroom'});
    const cx=(r.bounds_min[0]+r.bounds_max[0])/2,cz=(r.bounds_min[2]+r.bounds_max[2])/2;
    el('text',{x:mapPx(cx).toFixed(1),y:mapPz(cz).toFixed(1),'text-anchor':'middle',
      'dominant-baseline':'middle','class':'mlab'}).textContent=r.id;
  });
  (DATA.scene.doors||[]).forEach(d=>{            // edge = roomA → door → roomB
    const rs=(d.rooms||[]).map(id=>rooms.find(r=>r.id===id)).filter(Boolean);
    if(rs.length<2)return;
    const c=r=>[(r.bounds_min[0]+r.bounds_max[0])/2,(r.bounds_min[2]+r.bounds_max[2])/2];
    const pts=[c(rs[0])];
    if(d.obb)pts.push([d.obb.center[0],d.obb.center[2]]);
    pts.push(c(rs[1]));
    el('polyline',{points:pts.map(p=>mapPx(p[0]).toFixed(1)+','+mapPz(p[1]).toFixed(1)).join(' '),
      'class':'mdoor'});
    if(d.obb)el('circle',{cx:mapPx(d.obb.center[0]).toFixed(1),
      cy:mapPz(d.obb.center[2]).toFixed(1),r:2.2,'class':'mdot'});
  });
  let prev=null;                                  // trajectory, bold red
  const segPath={};
  frames.forEach(f=>{
    const k=f.segment||'';
    segPath[k]=(segPath[k]||'')+(prev&&prev.segment===f.segment?'L':'M')
      +mapPx(f.pos[0]).toFixed(1)+','+mapPz(f.pos[2]).toFixed(1);
    prev=f;});
  for(const k in segPath){
    const arc=k.startsWith('arc');
    const at={d:segPath[k],'class':arc?'marc':'mtraj'};
    if(arc)at.stroke=segColor[k]||'#8b93a3';
    el('path',at);
  }
  if(subj.length){let sp='';                      // subject walk path (dashed)
    subj.forEach((q,i)=>sp+=(i?'L':'M')+mapPx(q.pos[0]).toFixed(1)+','+mapPz(q.pos[2]).toFixed(1));
    el('path',{d:sp,'class':'msubj'});}
  mapCamG=el('g',{});
  const arr=el('line',{x2:0,y2:0,'class':'mcamlook'},mapCamG);
  el('circle',{r:3,'class':'mcam'},mapCamG);
  mapCamG._arr=arr;
})();
function updateMap(t){
  if(!mapCamG)return;
  const fr=frameAt(t);if(!fr)return;
  mapCamG.setAttribute('transform',
    `translate(${mapPx(fr.pos[0]).toFixed(1)},${mapPz(fr.pos[2]).toFixed(1)})`);
  const look=fr.look||[0,0,0],L=Math.hypot(look[0],look[2])||1;   // XZ heading
  mapCamG._arr.setAttribute('x2',(look[0]/L*11).toFixed(1));
  mapCamG._arr.setAttribute('y2',(look[2]/L*11).toFixed(1));
  let cur=null;                                   // room containing the camera
  for(const r of rooms)
    if(fr.pos[0]>=r.bounds_min[0]&&fr.pos[0]<=r.bounds_max[0]&&
       fr.pos[2]>=r.bounds_min[2]&&fr.pos[2]<=r.bounds_max[2]){cur=r.id;break;}
  if(cur!==mapCurRoom){
    if(mapCurRoom&&rcById[mapCurRoom])rcById[mapCurRoom].classList.remove('cur');
    if(cur&&rcById[cur])rcById[cur].classList.add('cur');
    mapCurRoom=cur;}
}
// ---------- point cloud (quantized base64 → Int16 triples) ----------
let CLOUD=null;
if(DATA.cloud){
  const bin=atob(DATA.cloud.b64),bytes=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++)bytes[i]=bin.charCodeAt(i);
  CLOUD={i16:new Int16Array(bytes.buffer),origin:DATA.cloud.origin,
         scale:DATA.cloud.scale,count:DATA.cloud.count,colors:[]};
  // vertical color range from the encoded min/max (scan once)
  let lo=1e9,hi=-1e9;
  for(let k=0;k<CLOUD.count;k++){const y=CLOUD.i16[k*3+1];
    if(y<lo)lo=y; if(y>hi)hi=y;}
  CLOUD.lo=CLOUD.origin[1]+lo*CLOUD.scale; CLOUD.hi=CLOUD.origin[1]+hi*CLOUD.scale;
  const stops=[[59,111,212],[89,196,160],[224,208,104]];
  for(let i=0;i<64;i++){const u=i/63*(stops.length-1),s=Math.floor(u),t=u-s,
    a=stops[s],b=stops[Math.min(s+1,stops.length-1)];
    CLOUD.colors.push(`rgb(${a.map((v,j)=>Math.round(v+(b[j]-v)*t)).join(',')})`);}
  if(DATA.cloud.c64){                                    // true-color u8 RGB
    const bin2=atob(DATA.cloud.c64),b2=new Uint8Array(bin2.length);
    for(let i=0;i<bin2.length;i++)b2[i]=bin2.charCodeAt(i);
    CLOUD.rgbU8=b2;CLOUD.rgb=[];for(let k=0;k<CLOUD.count;k++)
      CLOUD.rgb.push(`rgb(${b2[k*3]},${b2[k*3+1]},${b2[k*3+2]})`);
  }
}
// full (uncapped) cloud for 密 mode — same encoding, second payload
// full (uncapped) cloud for 密 mode — same encoding, second payload
let CLOUD_FULL=null;
function decodeCloud(c){const bin=atob(c.b64),by=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++)by[i]=bin.charCodeAt(i);
  const out={i16:new Int16Array(by.buffer),origin:c.origin,scale:c.scale,count:c.count,rgb:null};
  if(c.c64){const b2in=atob(c.c64),b2n=new Uint8Array(b2in.length);
    for(let i=0;i<b2in.length;i++)b2n[i]=b2in.charCodeAt(i);
    out.rgbU8=b2n;out.rgb=[];for(let k=0;k<out.count;k++)out.rgb.push(`rgb(${b2n[k*3]},${b2n[k*3+1]},${b2n[k*3+2]})`);}
  return out;}
if(DATA.cloud&&DATA.cloud.full)CLOUD_FULL=decodeCloud(DATA.cloud.full);
function currentCloud(){return (cloudModeAt===2&&CLOUD_FULL)?CLOUD_FULL:CLOUD;}

// ---------- WebGL point renderer (Canvas2D fallback if unavailable) ----------
// ?2d in the file URL forces the 2D path (debugging)
const FORCE_2D=location.search.indexOf('2d')>=0;
const GLV=`
attribute vec3 aPos;attribute vec3 aRGB;
uniform vec3 uOrigin,uEye,uR,uU,uF;uniform float uScale,uLo,uHi,uClean,uPt,uAspect;
varying vec3 vCol;
void main(){
  vec3 world=uOrigin+aPos*uScale;
  vec3 d=world-uEye;
  float xc=dot(d,uR),yc=dot(d,uU),zc=dot(d,uF);
  if(zc<0.05){gl_Position=vec4(0.,0.,2.,1.);gl_PointSize=0.;return;}
  // canvas2D px: W/2+0.9W·x/z (right+), H/2−0.9W·y/z (down+). GL ndc: right+ / UP+,
  // framebuffer y measured from bottom → ndc_y must KEEP yc's sign (no minus).
  gl_Position=vec4(1.8*xc, 1.8*uAspect*yc, (2.0*(zc-0.05)/59.95-1.0)*zc, zc);
  gl_PointSize=uPt;
  vec3 c1=vec3(0.231,0.435,0.831),c2=vec3(0.349,0.769,0.627),c3=vec3(0.878,0.816,0.408);
  float t=clamp((world.y-uLo)/max(uHi-uLo,1e-6),0.,1.)*2.;
  vec3 pal=t<1.?mix(c1,c2,t):mix(c2,c3,t-1.);
  vCol=uClean>0.5?aRGB:pal;
}`;
const GLF=`precision mediump float;varying vec3 vCol;
void main(){gl_FragColor=vec4(vCol,1.0);}`;
let GL=null;
function glInit(){
  if(FORCE_2D||!CLOUD)return;
  try{
    const cv=document.getElementById('glcv');
    const gl=cv.getContext('webgl',{alpha:false,antialias:true});
    if(!gl)return;
    const mk=(t,src)=>{const s=gl.createShader(t);gl.shaderSource(s,src);
      gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))
        throw new Error(gl.getShaderInfoLog(s));return s;};
    const pr=gl.createProgram();
    gl.attachShader(pr,mk(gl.VERTEX_SHADER,GLV));gl.attachShader(pr,mk(gl.FRAGMENT_SHADER,GLF));
    gl.linkProgram(pr);
    if(!gl.getProgramParameter(pr,gl.LINK_STATUS))throw new Error('link fail');
    gl.useProgram(pr);
    gl.enable(gl.DEPTH_TEST);
    const u=n=>gl.getUniformLocation(pr,n);
    const U={origin:u('uOrigin'),eye:u('uEye'),r:u('uR'),uu:u('uU'),f:u('uF'),
             scale:u('uScale'),lo:u('uLo'),hi:u('uHi'),clean:u('uClean'),
             pt:u('uPt'),aspect:u('uAspect')};
    const aPos=gl.getAttribLocation(pr,'aPos'),aRGB=gl.getAttribLocation(pr,'aRGB');
    const mkBuf=(arr)=>{const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
      gl.bufferData(gl.ARRAY_BUFFER,arr,gl.STATIC_DRAW);return b;};
    GL={gl,cv,U,aPos,aRGB,
        posB:mkBuf(CLOUD.i16),colB:mkBuf(CLOUD.rgbU8||new Uint8Array(CLOUD.count*3)),
        posF:CLOUD_FULL?mkBuf(CLOUD_FULL.i16):null,
        colF:CLOUD_FULL?mkBuf(CLOUD_FULL.rgbU8||new Uint8Array(CLOUD_FULL.count*3)):null};
  }catch(e){GL=null;}
}
function drawCloudGL(V){
  const {gl,cv,U,aPos,aRGB}=GL;
  cv.width=W;cv.height=H;gl.viewport(0,0,W,H);
  gl.clearColor(0.055,0.067,0.086,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  if(cloudModeAt===3)return;                         // 无 mode: cloud hidden
  const dense=cloudModeAt===2&&GL.posF;              // 密: full uncapped payload
  const src=dense?CLOUD_FULL:CLOUD;
  const step=dense?1:CLOUD_STEPS[cloudModeAt];
  const n=Math.floor((src.count-1)/step)+1;
  gl.uniform3fv(U.origin,src.origin);gl.uniform1f(U.scale,src.scale);
  gl.uniform3f(U.eye,V.ex,V.ey,V.ez);gl.uniform3fv(U.r,V.r);
  gl.uniform3fv(U.uu,V.u);gl.uniform3fv(U.f,V.f);
  gl.uniform1f(U.lo,CLOUD.lo);gl.uniform1f(U.hi,CLOUD.hi);
  gl.uniform1f(U.clean,cleanMode&&src.rgbU8?1:0);
  gl.uniform1f(U.pt,2.6*devicePixelRatio);gl.uniform1f(U.aspect,W/H);
  gl.bindBuffer(gl.ARRAY_BUFFER,dense?GL.posF:GL.posB);
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos,3,gl.SHORT,false,6*step,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,dense?GL.colF:GL.colB);
  gl.enableVertexAttribArray(aRGB);
  gl.vertexAttribPointer(aRGB,3,gl.UNSIGNED_BYTE,true,3*step,0);
  gl.drawArrays(gl.POINTS,0,n);
}
let showCloud=true,showLabels=true,cleanMode=false;
// cloud density 微/少/密/无: WebGL renders the FULL cloud comfortably, so
// default to 密 when it initialized; the 2D canvas fallback stays at 少
let cloudStep=CLOUD_STEPS[GL?2:1];
// ---------- canvas ----------
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
let W,H;
function resize(){W=cv.width=innerWidth*devicePixelRatio;H=cv.height=innerHeight*devicePixelRatio;
  cv.style.width=innerWidth+'px';cv.style.height=innerHeight+'px';}
addEventListener('resize',resize);resize();
let tCur=tMin,playing=false;
function quad(P,fill,stroke){if(!P||P.length<3)return;
  ctx.beginPath();ctx.moveTo(P[0].x,P[0].y);
  for(let i=1;i<P.length;i++)ctx.lineTo(P[i].x,P[i].y);ctx.closePath();
  if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();}}
function drawBox(corners,V,fill,stroke){
  const quads=FACES.map(f=>f.map(i=>project(corners[i],V,W,H))).filter(q=>q.every(p=>p));
  quads.sort((a,b)=>(a[0].z+a[2].z)-(b[0].z+b[2].z));
  quads.forEach(q=>quad(q,fill,stroke));
}
function line3(a,b,V,color,w,dash){
  const p=project(a,V,W,H),q=project(b,V,W,H);if(!p||!q)return;
  ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);
  ctx.strokeStyle=color;ctx.lineWidth=w||1.5;if(dash)ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);
}
function drawDot(p,color,r,V){const q=project(p,V,W,H);if(!q)return;
  ctx.beginPath();ctx.arc(q.x,q.y,r||4,0,7);ctx.fillStyle=color;ctx.fill();}
function drawCloud(V){
  if(!showCloud)return;
  const src=CLOUD;if(!src)return;                 // 2D fallback: base embed even
                                                  // under 密 (full would freeze CPU)
  const step=CLOUD_STEPS[cloudModeAt];
  if(!step)return;                                  // 无 mode: cloud hidden
  const {i16,origin,scale}=src;
  const lo=CLOUD?CLOUD.lo:0, hi=CLOUD?CLOUD.hi:1, span=Math.max(hi-lo,1e-6);
  for(let k=0;k<src.count;k+=step){
    const p=[origin[0]+i16[k*3]*scale,origin[1]+i16[k*3+1]*scale,origin[2]+i16[k*3+2]*scale];
    const q=project(p,V,W,H);if(!q)continue;
    const ci=Math.max(0,Math.min(63,Math.floor((p[1]-lo)/span*63)));
    ctx.fillStyle=(cleanMode&&src.rgb)?src.rgb[k]:CLOUD.colors[ci];
    ctx.fillRect(q.x-1,q.y-1,2.4,2.4);
  }
}
function drawLabel(p,text,color,V){
  const q=project(p,V,W,H);if(!q)return;
  ctx.font='11px -apple-system,"PingFang SC",sans-serif';
  const w=ctx.measureText(text).width,x=q.x+7,y=q.y-7;
  ctx.fillStyle='rgba(14,17,22,.72)';ctx.fillRect(x-3,y-12,w+6,16);
  ctx.fillStyle=color;ctx.fillText(text,x,y);
}
function drawMarker(p,color,V){
  const q=project(p,V,W,H);if(!q)return;
  ctx.strokeStyle=color;ctx.lineWidth=1.6;ctx.beginPath();
  ctx.moveTo(q.x-4,q.y);ctx.lineTo(q.x+4,q.y);
  ctx.moveTo(q.x,q.y-4);ctx.lineTo(q.x,q.y+4);ctx.stroke();
}
function drawLabels(V){
  if(!showLabels)return;
  objs.forEach(({o})=>{                       // object centers
    drawMarker(o.obb.center,'rgba(160,190,240,.9)',V);
    drawLabel(o.obb.center,`${o.label}`,'#a9c2f5',V);
  });
  Object.entries(DATA.anchors||{}).forEach(([tgt,a])=>{   // anchor picks
    drawMarker(a.pos,'#ffe14d',V);
    drawLabel(a.pos,`anchor ${tgt} (${(a.score||0).toFixed(2)})`,'#ffe14d',V);
  });
}
function drawFrustum(pos,look,fovDeg,V,len,fill,stroke,rayColor){
  // camera view frustum: apex at pos, 16:9 image rect at depth len along the
  // look direction, sized by the vertical FOV (fov_deg per frame, default 55)
  const fov=fovDeg||55, t=Math.tan(fov*Math.PI/360), aspect=16/9;
  const dir=norm3(sub3(look,pos));
  let up=[0,1,0]; if(Math.abs(dot3(dir,up))>0.99)up=[1,0,0];
  const right=norm3(cross3(dir,up)), uv=cross3(right,dir);
  const c=add3(pos,mul3(dir,len)), hh=len*t, hw=len*t*aspect;
  const rc=[[hh,hw],[hh,-hw],[-hh,-hw],[-hh,hw]].map(([a,b])=>
    add3(add3(c,mul3(uv,a)),mul3(right,b))).map(p=>project(p,V,W,H));
  const apex=project(pos,V,W,H);
  if(!apex||rc.some(p=>!p))return;
  rc.forEach(p=>{ctx.beginPath();ctx.moveTo(apex.x,apex.y);ctx.lineTo(p.x,p.y);
    ctx.strokeStyle=rayColor;ctx.lineWidth=1;ctx.stroke();});
  quad(rc,fill,stroke);
}
let GL_TRIED=false;
function drawView(t){
  ctx.clearRect(0,0,W,H);const V=viewMatrix();
  // point cloud first: WebGL when available, 2D software path otherwise
  if(!GL&&!GL_TRIED){GL_TRIED=true;glInit();}
  if(GL){
    if(showCloud)drawCloudGL(V);
    else{const gl=GL.gl;GL.cv.width=W;GL.cv.height=H;
      gl.clearColor(0.055,0.067,0.086,1);
      gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);}   // P hides the cloud
  }else drawCloud(V);
  if(!cleanMode){
  // rooms (AABB wireframes)
  rooms.forEach(r=>{
    const a=r.bounds_min,b=r.bounds_max;
    const cs=obbCorners({center:[(a[0]+b[0])/2,(a[1]+b[1])/2,(a[2]+b[2])/2],
      rotation:[[1,0,0],[0,1,0],[0,0,1]],extents:[b[0]-a[0],b[1]-a[1],b[2]-a[2]]});
    drawBox(cs,V,null,'rgba(120,140,170,.25)');
  });
  // structures (walls/floor/ceiling): translucent gray
  structs.forEach(({o,c})=>drawBox(c,V,'rgba(140,150,165,.10)','rgba(160,170,190,.35)'));
  // doors: green
  doors.forEach(({o,c})=>drawBox(c,V,'rgba(70,200,130,.22)','rgba(90,230,150,.7)'));
  // objects
  objs.forEach(({o,c})=>drawBox(c,V,'rgba(90,120,180,.20)','rgba(130,160,220,.8)'));
  }
  // subject track
  if(subj.length){
    for(let i=0;i<subj.length-1;i++)
      line3(addY(subj[i].pos,0.9),addY(subj[i+1].pos,0.9),V,'rgba(255,190,80,.7)',2);
    const s=subjectAt(t);if(s){drawCapsule(s,V,t);}
  }
  // trajectory: actual shots in bold RED; 1-3-1 transition arcs stay thin in
  // their segment color (they are connectors, not footage)
  let prev=null;
  frames.forEach(f=>{
    const arc=f.segment.startsWith('arc');
    const col=f.colliding?'#ff5555':(f.occluded?'#ffaa33':
      (arc?(segColor[f.segment]||'#8b93a3'):'#ff3232'));
    if(prev&&prev.segment===f.segment)
      line3(prev.pos,f.pos,V,col,f.colliding?4:(arc?1.6:4.5));
    prev=f;
  });
  // collision / occlusion dots
  frames.forEach(f=>{if(f.colliding)drawDot(f.pos,'#ff5555',4.5,V);
                     else if(f.occluded)drawDot(f.pos,'#ffaa33',3.5,V);});
  // ground-truth cloud clearance (OBB field can miss uncovered geometry):
  // magenta <0.10 m (grazing real points), light violet <0.25 m
  frames.forEach(f=>{const c=f.cloud_m;
    if(c==null||f.colliding)return;
    if(c<0.10)drawDot(f.pos,'#ff4dd2',3.6,V);
    else if(c<0.25)drawDot(f.pos,'#b892ff',2.6,V);});
  // ghost frustums along the path (static orientation cues, segment-colored)
  const gStep=Math.max(1,Math.floor(frames.length/10));
  frames.forEach((f,i)=>{
    if(i%gStep&&i!==frames.length-1)return;
    const col=segColor[f.segment]||'#5b8cff';
    drawFrustum(f.pos,f.look,f.fov_deg,V,0.85,
      hexA(col,.06),hexA(col,.5),hexA(col,.35));
  });
  // live camera frustum at t
  const fr=frameAt(t);
  if(fr){
    drawFrustum(fr.pos,fr.look,fr.fov_deg,V,1.8,
      'rgba(255,255,255,.10)','#ffffff','rgba(255,255,255,.55)');
    drawDot(fr.pos,'#ffffff',5,V);
    line3(fr.pos,fr.look,V,'rgba(255,255,255,.30)',1,[4,4]);
  }
  // labeled anchors + object centers on top
  if(!cleanMode)drawLabels(V);
}
function subjectAt(t){
  let a=subj[0],b=subj[subj.length-1];
  for(let i=0;i<subj.length-1;i++)if(t>=subj[i].t&&t<=subj[i+1].t){a=subj[i];b=subj[i+1];break;}
  if(b.t<=a.t)return {pos:a.pos,heading:a.heading};
  const u=(t-a.t)/(b.t-a.t);
  return {pos:[a.pos[0]+(b.pos[0]-a.pos[0])*u,a.pos[1]+(b.pos[1]-a.pos[1])*u,a.pos[2]+(b.pos[2]-a.pos[2])*u],
          heading:a.heading+(b.heading-a.heading)*u};
}
function drawCapsule(s,V,t){
  const h=(DATA.scene.subject_track.height_m||1.75);
  const a=[s.pos[0],s.pos[1]+0.15,s.pos[2]],b=[s.pos[0],s.pos[1]+0.9*h,s.pos[2]];
  line3(a,b,V,'#ffbe50',5);
  drawDot(b,'#ffbe50',6,V);
  // heading arrow
  const d=[Math.cos(s.heading),0,Math.sin(s.heading)];
  line3(b,add3(b,mul3(d,0.5)),V,'rgba(255,190,80,.8)',2);
}
function frameAt(t){if(!frames.length)return null;
  let best=frames[0];for(const f of frames)if(Math.abs(f.t-t)<Math.abs(best.t-t))best=f;return best;}
// ---------- interaction ----------
let drag=null;
cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY};cv.style.cursor='grabbing';});
addEventListener('mousemove',e=>{if(!drag)return;
  cam.yaw+=(e.clientX-drag.x)*0.008;cam.pitch=Math.max(-1.4,Math.min(1.4,cam.pitch+(e.clientY-drag.y)*0.008));
  drag={x:e.clientX,y:e.clientY};dirty=true;});
addEventListener('mouseup',()=>{drag=null;cv.style.cursor='grab';});
cv.addEventListener('wheel',e=>{cam.dist=Math.max(2,Math.min(60,cam.dist*(1+e.deltaY*0.001)));dirty=true;},{passive:true});
const slider=document.getElementById('time'),tlabel=document.getElementById('tlabel');
slider.addEventListener('input',()=>{tCur=tMin+(tMax-tMin)*slider.value/1000;});
addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();togglePlay();}
  else if(e.key==='p'||e.key==='P'){showCloud=!showCloud;dirty=true;}
  else if(e.key==='l'||e.key==='L'){showLabels=!showLabels;dirty=true;}
  else if(e.key==='b'||e.key==='B'){setCloudMode((cloudModeAt+1)%4);}
  else if(e.key==='c'||e.key==='C'){toggleClean();}});
let cloudModeAt=1;
function setCloudMode(m){cloudModeAt=m;cloudStep=CLOUD_STEPS[m];
  document.querySelectorAll('.bx').forEach(b=>{
    b.style.outline=+b.dataset.m===m?'2px solid #5b8cff':'none';});dirty=true;}
document.querySelectorAll('.bx').forEach(b=>
  b.addEventListener('click',()=>setCloudMode(+b.dataset.m)));
document.getElementById('play').addEventListener('click',togglePlay);
function toggleClean(){cleanMode=!cleanMode;
  document.getElementById('clean').style.outline=cleanMode?'2px solid #5b8cff':'none';
  dirty=true;}
document.getElementById('clean').addEventListener('click',toggleClean);
function togglePlay(){playing=!playing;document.getElementById('play').textContent=playing?'⏸':'▶';
  if(playing&&tCur>=tMax-1e-3)tCur=tMin;}
let last=performance.now(),dirty=true;
const markDirty=()=>{dirty=true;};
cv.addEventListener('mousedown',markDirty);
addEventListener('resize',markDirty);
slider.addEventListener('input',markDirty);
setCloudMode(1);   // after `dirty` exists — TDZ-safe init of the default (少)
function tick(now){
  if(playing){tCur+=(now-last)/1000;if(tCur>tMax){tCur=tMax;playing=false;
    document.getElementById('play').textContent='▶';}
    slider.value=1000*(tCur-tMin)/(tMax-tMin);dirty=true;}
  last=now;
  if(dirty){drawView(tCur);dirty=false;
    updateMap(tCur);
    tlabel.textContent='t='+tCur.toFixed(1)+'s';}
  requestAnimationFrame(tick);
}
// fit camera to scene: grow dist until the whole content bbox projects
// inside 4–96% × 6–90% of the viewport (bottom band reserved for the bar)
(function(){
  let lo=[1e9,1e9,1e9],hi=[-1e9,-1e9,-1e9];
  const put=p=>{for(let i=0;i<3;i++){lo[i]=Math.min(lo[i],p[i]);hi[i]=Math.max(hi[i],p[i]);}};
  rooms.forEach(r=>{put(r.bounds_min);put(r.bounds_max);});
  if(frames.length)frames.forEach(f=>put(f.pos));
  if(CLOUD){put(CLOUD.origin);
    put(CLOUD.origin.map(v=>v+32767*CLOUD.scale));}
  cam.target=[(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,(lo[2]+hi[2])/2];
  const corners=[];for(const x of[lo[0],hi[0]])for(const y of[lo[1],hi[1]])
    for(const z of[lo[2],hi[2]])corners.push([x,y,z]);
  for(cam.dist=6;cam.dist<120;cam.dist+=0.5){
    const V=viewMatrix();
    if(corners.every(c=>{const q=project(c,V,W,H);
      return q&&q.x>W*0.04&&q.x<W*0.96&&q.y>H*0.06&&q.y<H*0.90;}))break;
  }
})();
// HUD
document.getElementById('metrics').innerHTML=Object.entries(DATA.metrics||{}).map(
  ([k,v])=>`<span>${k}</span><b>${v}</b>`).join('');
document.getElementById('legend').innerHTML=
  segNames.map(n=>`<div><span class="sw" style="background:${segColor[n]}"></span>${segDisp(n)}</div>`).join('')
  +'<div><span class="sw" style="background:#ffe14d"></span>anchor</div>'
  +'<div><span class="sw" style="background:#ff3232"></span>运镜轨迹(红·过渡弧除外)</div>'
  +'<div><span class="sw" style="background:#a9c2f5"></span>object center</div>'
  +'<div><span class="sw" style="background:#59c4a0"></span>点云(按高度)</div>'
  +'<div><span class="sw" style="background:#ff5555"></span>collision</div>'
  +'<div><span class="sw" style="background:#ff4dd2"></span>云净空&lt;0.10m</div>'
  +'<div><span class="sw" style="background:#b892ff"></span>云净空&lt;0.25m</div>'
  +'<div><span class="sw" style="background:#ffaa33"></span>occluded</div>'
  +'<div><span class="sw" style="background:#ffbe50"></span>subject</div>';
requestAnimationFrame(tick);
</script>
</body>
</html>"""


def build_html(scene_dict: dict, traj_final: dict, anchors: dict | None = None,
               cloud: dict | None = None) -> str:
    data = {
        "scene": scene_dict,
        "frames": traj_final.get("frames", []),
        "metrics": traj_final.get("metrics", {}),
        "anchors": anchors or {},
        "cloud": cloud,
    }
    html = _TEMPLATE.replace("__SCENE_ID__", scene_dict.get("scene_id", "scene"))
    html = html.replace("__CLOUD_STEPS__", json.dumps(list(CLOUD_MODE_STEPS)))
    return html.replace("__DATA__", json.dumps(data, ensure_ascii=False))


def run(cfg=None, traj_path=None, out_path=None, scene_path=None,
        force=False, cloud_voxel=None, max_points=None,
        cloud_margin=None) -> str:
    cfg = cfg or load_config()
    ct = cfg.get("cinema_traj", {})
    traj_path = Path(traj_path) if traj_path else None
    if traj_path is None:
        scene_id = Path(ct.get("scene", "demo_apartment.json")).stem
        base = Path(ct.get("outputs_dir", "cinema_traj_out"))
        if not base.is_absolute():
            from vlm_intention.common.config import repo_root
            base = repo_root() / base
        traj_path = base / scene_id / "traj_final.json"
    traj = load_json(traj_path)
    if not traj.get("frames"):
        raise ValueError(f"{traj_path} has no frames — run the optimize stage first")

    # scene source: explicit CLI arg wins over the config default (the config
    # scene may point at another scene → silent fallback with no geometry)
    scene_file = Path(scene_path) if scene_path else Path(ct.get("scene", ""))
    if scene_file.name and not scene_file.is_absolute() and not scene_file.exists():
        from vlm_intention.common.config import repo_root
        scene_file = repo_root() / scene_file
    scene_dict = load_json(scene_file) if scene_file.exists() else \
        {"scene_id": traj.get("scene_id", "scene")}
    # config scene must match the trajectory's scene, else rooms/labels lie
    if scene_dict.get("scene_id") != traj.get("scene_id"):
        scene_dict = {"scene_id": traj.get("scene_id", "scene")}

    # anchor picks live in traj_init.json next to traj_final.json
    anchors = {}
    init_path = traj_path.parent / "traj_init.json"
    if init_path.exists():
        anchors = load_json(init_path).get("anchors") or {}

    cloud = _load_source_cloud(
        scene_dict,
        max_pts=int(max_points) if max_points is not None else CLOUD_MAX_POINTS,
        voxel=float(cloud_voxel) if cloud_voxel is not None else CLOUD_VOXEL,
        margin=1.5 if cloud_margin is None else float(cloud_margin))

    out_path = Path(out_path) if out_path else traj_path.parent / "viewer" / \
        f"{traj.get('scene_id', 'scene')}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_near, n_hit = _annotate_cloud_clearance(traj.get("frames", []), cloud)
    out_path.write_text(build_html(scene_dict, traj, anchors, cloud),
                        encoding="utf-8")
    extra = f" · 点云 {cloud['count']} 点" if cloud else ""
    if cloud:
        extra += f" · 云净空: {n_hit} 帧<0.10m / {n_near} 帧<0.25m"
    print(f"👁️  viewer → {out_path}{extra}")
    return str(out_path)


CLOUD_HIT_M = 0.10            # drone-level: path grazes real geometry
CLOUD_NEAR_M = 0.25           # visually tight — worth showing


def _annotate_cloud_clearance(frames: list, cloud: dict | None) -> tuple:
    """Per-frame distance to the nearest REAL cloud point (f.cloud_m).

    The optimizer's collision field is the scene-graph OBB union — geometry
    the boxes don't cover (thin decor, details, 3DGS fuzz) reads as free
    while the camera actually grazes points. This ground-truth overlay is
    computed against the same packed cloud the viewer renders, so what you
    see is what was measured. Returns (n_near, n_hit)."""
    if not cloud or not frames:
        return 0, 0
    try:
        import base64

        q = np.frombuffer(base64.b64decode(cloud["b64"]), dtype="<i2").reshape(-1, 3)
        pts = q * cloud["scale"] + np.asarray(cloud["origin"])
        pos = np.asarray([f["pos"] for f in frames], dtype=np.float64)
        # (F,1,3) − (1,N,3) in chunks to bound memory for big clouds
        dmin = np.full(len(pos), np.inf)
        step = max(1, 4_000_000 // max(len(pts), 1))
        for i in range(0, len(pos), step):
            chunk = pos[i:i + step]
            d = np.linalg.norm(chunk[:, None, :] - pts[None, :, :], axis=2)
            dmin[i:i + step] = d.min(axis=1)
        for f, d in zip(frames, dmin):
            f["cloud_m"] = round(float(d), 3)
        return int((dmin < CLOUD_NEAR_M).sum()), int((dmin < CLOUD_HIT_M).sum())
    except Exception as e:                                # noqa: BLE001
        print(f"   ⚠️  云净空计算失败（忽略该层）: {e}")
        return 0, 0


def _load_source_cloud(scene_dict: dict, max_pts: int = CLOUD_MAX_POINTS,
                       voxel: float = CLOUD_VOXEL, margin: float = 1.5) -> dict | None:
    """Pack the scene's source map (provenance.source_map) for embedding.

    The hand-refined scene may live in a shifted frame (e.g. floor moved to
    y=0); provenance.map_shift [dx,dy,dz] is applied when present, otherwise
    the cloud is auto-aligned vertically by matching its floor-histogram mode
    to the scene room floor. Best-effort: any failure just disables the
    cloud layer.
    """
    prov = scene_dict.get("provenance") or {}
    src = prov.get("source_map", "")
    if not src:
        return None
    p = Path(src)
    if not p.is_absolute():
        p = Path(_REPO_ROOT) / p
    if not p.exists():
        return None
    try:
        from cinema_traj.stages.extract import (_to_yup, detect_floor_ceiling,
                                                load_points, voxel_downsample)

        pts = np.asarray(load_points(str(p)), dtype=np.float64)
        cols = _read_gs_colors(p)                    # u8 per raw vertex, if 3DGS PLY
        pts = _to_yup(pts, up=str(prov.get("map_up", "y")))   # declared convention
        shift = list(prov.get("map_shift") or [0.0, 0.0, 0.0])
        if not shift[1] and scene_dict.get("rooms"):
            scene_floor = float(scene_dict["rooms"][0]["bounds_min"][1])
            cloud_floor, _ = detect_floor_ceiling(voxel_downsample(pts, 0.10))
            shift[1] = scene_floor - cloud_floor
        pts[:, 0] += shift[0]; pts[:, 1] += shift[1]; pts[:, 2] += shift[2]
        if scene_dict.get("rooms") and margin >= 0:     # clip outside context
            lo = np.asarray(scene_dict["rooms"][0]["bounds_min"]) - margin
            hi = np.asarray(scene_dict["rooms"][0]["bounds_max"]) + margin
            inside = np.all((pts >= lo) & (pts <= hi), axis=1)
            pts = pts[inside]
            if cols is not None:
                cols = cols[inside]
        base = pack_point_cloud(pts, max_pts=max_pts, voxel=voxel, colors=cols)
        if base is not None:
            # 密 mode payload: same cloud, NO point cap (voxel-only drop)
            full = pack_point_cloud(pts, max_pts=None, voxel=voxel, colors=cols)
            base["full"] = full
        return base
    except Exception as e:                                # noqa: BLE001
        print(f"   ⚠️  点云加载失败（viewer 将不带点云层）: {e}")
        return None


if __name__ == "__main__":                                # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="生成离线 3D 轨迹查看器 HTML")
    ap.add_argument("--traj", help="traj_final.json 路径")
    ap.add_argument("--out", help="输出 HTML 路径")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else None
    run(cfg, traj_path=a.traj, out_path=a.out)
