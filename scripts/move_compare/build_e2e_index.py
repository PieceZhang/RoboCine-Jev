#!/usr/bin/env python3
"""E2E INDEX: matrix of actual videos + wall-clock timing per move/side."""
import subprocess
from pathlib import Path

E2E = Path("/home/w84384651/Documents/RoboCine-Siyu/cinema_traj_out/move_compare_e2e")
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


def side_info(m, side):
    d = E2E / m / side
    f = d / "status.txt"
    if not f.exists():
        return None
    st = {}
    for line in f.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            st[k] = v
    if "move" not in st:                      # marker row (FAIL / N/A note)
        return {"marker": f.read_text().strip().splitlines()[0]}
    vid = d / "actual_video.mp4"
    return {
        "rc": st.get("rc", "?"),
        "wall": st.get("wall_s", "?"),
        "wbc_pass": st.get("wbc_pass", "0"),
        "video": vid.exists(),
        "vdur": dur(vid) if vid.exists() else "-",
    }


rows = []
for m in MOVES:
    row = {"move": m}
    for side in ("baseline", "fm"):
        row[side] = side_info(m, side)
    rows.append(row)


def cell(s):
    if s is None:
        return "—（未运行）"
    if "marker" in s:
        return "**%s**" % s["marker"]
    vid = "✅ 实拍 %s" % s["vdur"] if s["video"] else "❌ 无视频"
    wbc = "WBC ✅" if s["wbc_pass"] not in ("0", "") else "WBC ❌"
    return "%s · %s · %ss" % (vid, wbc, s["wall"])


lines = ["# PR #28 E2E 对比矩阵 — 控制器执行 + 末端相机实拍", "",
         "目录: `%s`" % E2E,
         "每格: actual_video.mp4（机器人末端相机实拍 ros_capture/fp_cam.mp4）· WBC 跟踪 "
         "PASS(<3cm mean) · 全链 wall 时长。run.log / step_timing.log 同目录。", "",
         "| move | baseline (E2E 实拍) | fm (E2E 实拍) |",
         "|---|---|---|"]
for r in rows:
    lines.append("| %s | %s | %s |" % (r["move"],
                                       cell(r["baseline"]) if r["baseline"] else "—（未运行）",
                                       cell(r["fm"]) if r["fm"] else "—（未运行）"))
n_vid = sum(1 for r in rows for s in (r["baseline"], r["fm"]) if s and s.get("video"))
lines += ["", "实拍视频共 %d 个（目标 baseline 12 + fm 8 = 20）。" % n_vid, ""]
for r in rows:
    for side in ("baseline", "fm"):
        s = r[side]
        if s and s.get("video"):
            lines.append("- %s/%s: `%s` (%s, wall %ss)" % (
                r["move"], side, E2E / r["move"] / side / "actual_video.mp4",
                s["vdur"], s["wall"]))
(E2E / "E2E_INDEX.md").write_text("\n".join(lines) + "\n")
print("WROTE", E2E / "E2E_INDEX.md", "videos:", n_vid)
