#!/usr/bin/env python3
"""move_compare · planner E2E timing consolidation (R1 / R2 / fast) → TIMING.fast.md.

Sources
- R1 (9月23 matrix, from the recorded baseline screenshot; pre disc-candidate,
  pre crane-window fix): embedded below as ROUND1 — static historical reference.
- R2 (9月24 morning, `*.baseline.plan.log`, 12-step multi-process chain with
  sequential ik_stream): parsed live from the logs.
- fast (`*.fast.plan.log`, produced by rerun_fast.sh via tools/plan_fast.py):
  parsed live from the logs; E2E_WALL is the /usr/bin/time wall clock.

Usage: python3 scripts/move_compare/build_timing_table.py [move_compare_dir]
"""
import re
import sys
from pathlib import Path

def _default_dir():
    """<repo>/cinema_traj_out/move_compare when installed in scripts/move_compare."""
    try:
        return Path(__file__).resolve().parents[2] / "cinema_traj_out" / "move_compare"
    except IndexError:
        return None


STEPS = ["plan+build+optimize+audit", "export", "retime (jarvis)",
         "view (html)", "simview build"]

# R1 · 9月23 矩阵（截图存档）: variant -> (12步合计 s, 备注)
ROUND1 = {
    "dolly_in":     ("67.9", "step3 59.6s 离群（10×）"),
    "dolly_out":    ("16.0", ""),
    "pan_left":     ("14.0", ""),
    "pan_right":    ("14.0", ""),
    "orbit_swing":  ("14.5", ""),
    "static_shot":  ("14.1", ""),
    "zoom_in":      ("14.1", "R1 仅 zoom"),
    "zoom_out":     ("14.1", "R1 仅 zoom"),
    "truck_left":   ("—", "R1 未测"),
    "truck_right":  ("—", "R1 未测"),
    "crane_up":     ("FAIL", "可行高度带拒绝"),
    "crane_down":   ("FAIL", "可行高度带拒绝"),
}
# FM planner · 9月23 fm_report 合计（秒，✅/❌ = 审计）
FM = {
    "dolly_in": "2.4✅", "dolly_out": "3.7✅",
    "pan_left": "6.9✅", "pan_right": "7.4✅",
    "static_shot": "5.3✅", "zoom_in": "5.3✅", "zoom_out": "5.3✅",
    "orbit_swing": "12.8❌",
    "truck_left": "—", "truck_right": "—",
    "crane_up": "FAIL", "crane_down": "FAIL",
}


def parse_round2(log: Path):
    """12-step chain: return (step3/export/retime breakdown, sum of 5 steps)."""
    txt = log.read_text(errors="replace")
    vals = {}
    for st in STEPS:
        m = re.search(rf"{re.escape(st)}[^:\n]*: ([0-9.]+)s", txt)
        if m:
            vals[st] = float(m.group(1))
    if not vals:
        return "—", "—"
    s3 = vals.get("plan+build+optimize+audit", 0.0)
    ex = vals.get("export", 0.0)
    rt = vals.get("retime (jarvis)", 0.0)
    return f"{s3:.1f} / {ex:.1f} / {rt:.1f}", f"{sum(vals.values()):.1f}"


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_dir()
    if d is None:
        sys.exit("无法推断 move_compare 目录（脚本未安装在 <repo>/scripts/move_compare/）；"
                 "请显式传参: build_timing_table.py <move_compare_dir>")
    rows = []
    for f in sorted(d.glob("*.fast.plan.log")):
        v = f.name.replace(".fast.plan.log", "")
        txt = f.read_text(errors="replace")
        e2e = re.search(r"E2E_WALL=([0-9.]+)s", txt)
        marks = len(re.findall(r"⏱ \[fast", txt))
        ok = "check_trackable ✅" in txt
        r2s, r2t = parse_round2(d / f"{v}.baseline.plan.log")
        r1, note = ROUND1.get(v, ("—", ""))
        fast = f"{float(e2e.group(1)):.2f}" if e2e else "FAIL"
        try:
            sp = f"{float(r2t) / float(fast):.1f}×"
        except (ValueError, ZeroDivisionError):
            sp = "—"
        flag = "✅" if (e2e and float(e2e.group(1)) < 5.0
                       and marks >= 7 and ok) else "⚠️"
        rows.append((v, r1, FM.get(v, "—"), r2s, r2t, fast, sp, note, flag))

    L = ["# move_compare · planner E2E 耗时总表（单运镜）", "",
         "- **R1 基线** = 9月23 矩阵（12 步多进程链；圆盘候选枚举与 crane 可行窗修复之前。"
         "dolly_in step3 离群 10×、truck 未测、crane 全 FAIL）——截图存档值",
         "- **FM** = 9月23 FM planner 合计（❌ = 审计未过）——截图存档值",
         "- **R2 基线** = 9月24 上午 12 步链（core 补丁已在树，export ik_stream 仍为顺序版"
         "→ truck/orbit 的 retime 离群）；三列为 step3 / export / retime，合计为 5 个重步之和",
         "- **fast** = `scripts/move_compare/rerun_fast.sh` → `tools/plan_fast.py` 单进程全链"
         "（E2E_WALL 实测墙钟；7 marks = fast 路径完整走完 + check_trackable ✅）",
         "", "| 运镜 | R1 基线 | FM | R2 step3/exp/retime | R2 合计 | **fast E2E** | R2→fast | 备注 | <5s |",
         "|---|---|---|---|---|---|---|---|---|"]
    for v, r1, fm, r2s, r2t, fast, sp, note, flag in rows:
        L.append(f"| {v} | {r1} | {fm} | {r2s} | {r2t} | **{fast}** | {sp} | {note} | {flag} |")
    n_ok = sum(1 for r in rows if r[8] == "✅")
    worst = max((float(r[5]) for r in rows if r[5] != "FAIL"), default=0.0)
    L += ["", f"**fast 全部 <5s：{n_ok}/{len(rows)} ✅**（最慢 {worst:.2f}s）"]

    out = d / "TIMING.fast.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
