#!/usr/bin/env python3
"""Round-2 comparison INDEX: 12 moves, fm N/A rows, notes."""
import json
import subprocess
from pathlib import Path

CMP = Path("/home/w84384651/Documents/RoboCine-Siyu/cinema_traj_out/move_compare")
MOVES = ["dolly_in", "dolly_out", "pan_left", "pan_right", "orbit_swing",
         "crane_up", "crane_down", "static_shot", "zoom_in", "zoom_out",
         "truck_left", "truck_right"]


def dur(p):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(p)], capture_output=True, text=True)
        return "%.1fs" % float(r.stdout.strip())
    except Exception:
        return "?"


def cell(side):
    if side.get("na"):
        return "N/A（fm 词汇表无 truck 类型）"
    if side["failed"]:
        return "**FAIL**"
    h = "Y" if side["html"] else "N"
    return "%s / %s / %s" % (h, side["shot"] or "-", side["overview"] or "-")


rows = []
for m in MOVES:
    row = {"move": m}
    for side in ("baseline", "fm"):
        d = CMP / m / side
        row[side] = {
            "html": (d / "viewer.html").exists(),
            "shot": dur(d / "shot.mp4") if (d / "shot.mp4").exists() else None,
            "overview": dur(d / "overview.mp4") if (d / "overview.mp4").exists() else None,
            "failed": (d / "FAILED").exists(),
            "na": (d / "UNSUPPORTED").exists(),
        }
    rep = CMP / m / "fm_run" / "fm_report.json"
    if rep.exists():
        try:
            r = json.loads(rep.read_text())
            a = r.get("audit", {})
            mt = a.get("metrics", {})
            row["fm_report"] = "viol %.3f coll %s reach %s" % (
                a.get("violation_rate", -1), mt.get("collision_rate", "?"),
                mt.get("reachable_frac", "?"))
        except Exception:
            row["fm_report"] = "parse-err"
    rows.append(row)

lines = ["# PR #28 运镜对比测试矩阵（round 2 — 用户反馈修正版）", "",
         "目录: `%s`" % CMP,
         "N=%d 运镜 x (baseline | fm) x (viewer.html, shot.mp4 第一人称, overview.mp4 第三人称)" % len(MOVES),
         "本轮修正：dolly_in 换 SE 接近走廊（1.6m 运动）；pan 单向（sweep_return:false）；"
         "orbit 基线改真弧线（弓形 alpha）；crane 用实测可达高度窗 [0.55,0.63]（8cm 幅度）；"
         "zoom 拆 in/out；新增 truck_left/right；fm 渲染带 cast（修复黄胶囊）；fm truck=N/A。", "",
         "| move | baseline (html/shot/overview) | fm (html/shot/overview) | fm 审计 |",
         "|---|---|---|---|"]
for r in rows:
    rep = r.get("fm_report", "")
    cb = cell(r["baseline"]).replace(" / ", " | ")
    cf = cell(r["fm"]).replace(" / ", " | ")
    lines.append("| %s | %s | %s | %s |" % (r["move"], cb, cf, rep[:60]))

lines += ["", "## 发现（round 2）", "",
          "- **高度-俯仰耦合实测**：正南走廊以头部位姿为目标时，相机高度仅 [0.55,0.63] 可达"
          "（物理带标称 [0.25,0.63]）——P2（俯仰-高度查表）的实证数据；crane 幅度因此为 8cm",
          "- **orbit**：基线=弓形弧线（alpha 锁定使 SwingSegment 过中方位圆点）；"
          "fm=单段 pan 形式（fm 分支无螺旋修复、orbit_quarter 不可达，多段 posts 已按用户要求弃用）",
          "- **truck**：pan/truck 语义分离（6ad167a）后基线支持横移 0.93m；fm 词汇表无该类型 → N/A",
          "- **dolly_in**：默认正面接近被茶几/底座泊位钳死（0 运动）；azimuth_deg=60 的 SE 走廊 1.6m 可行",
          ""]

for r in rows:
    lines.append("### %s" % r["move"])
    for side in ("baseline", "fm"):
        p = CMP / r["move"] / side
        if (p / "viewer.html").exists():
            lines.append("- **%s**: `%s/viewer.html` · `%s/shot.mp4` · `%s/overview.mp4`" % (side, p, p, p))
        elif (p / "FAILED").exists():
            lines.append("- **%s**: FAILED（规划失败，无产物）" % side)
        elif (p / "UNSUPPORTED").exists():
            lines.append("- **%s**: N/A（fm 词汇表不支持）" % side)
    lines.append("")

(CMP / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[7:19]))
fails = [(r["move"], s) for r in rows for s in ("baseline", "fm") if r[s]["failed"]]
nas = [(r["move"]) for r in rows if r["fm"]["na"]]
missing = [(r["move"], s) for r in rows for s in ("baseline", "fm")
           if not r[s]["failed"] and not r[s]["na"] and not (CMP / r["move"] / s / "shot.mp4").exists()]
print("FAILED:", fails)
print("FM N/A:", nas)
print("MISSING videos:", missing if missing else "none")
