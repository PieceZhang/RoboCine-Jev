# move_compare · 单运镜 planner E2E 耗时总表

三轮测量汇总（kujiale_0021 场景 · 每运镜独立 yml · 5090-jarvis）。
快档链路 `tools/plan_fast.py`（PR #34）后 **11/12 运镜 E2E < 5s**（dolly_in 11.5s 例外：#36 走廊旋转回退在首选射线不可行时追加可达性扫描）。

| 轮次 | 日期 | 链路 | 说明 |
|---|---|---|---|
| R1 基线 | 9月23 | 12 步多进程链 | 圆盘候选枚举与 crane 可行窗修复之前（截图存档值） |
| FM | 9月23 | FM planner 替代 plan+build+optimize | 审计门兜底；❌ = 审计未过 |
| R2 基线 | 9月24 上午 | 12 步多进程链（core 补丁已在树） | export ik_stream 仍为顺序版 → truck 的 retime、orbit 的 export+retime 离群 |
| fast | 9月24 | `scripts/move_compare/rerun_fast.sh` → `tools/plan_fast.py` | 单进程全链，E2E_WALL 实测墙钟，7 marks + check_trackable ✅ |

| 运镜 | R1 基线 (s) | FM (s) | R2 step3/exp/retime | R2 合计 (s) | fast E2E (s) | R2→fast | 备注 | <5s |
|---|---|---|---|---|---|---|---|---|
| crane_down | FAIL | FAIL | 5.6 / 2.3 / 2.6 | 13.7 | **3.51** | 3.9× | R1 可行高度带拒绝；可行窗修复后 R2 起可规划 | ✅ |
| crane_up | FAIL | FAIL | 5.3 / 2.2 / 2.5 | 14.3 | **3.51** | 4.1× | 同上 | ✅ |
| dolly_in | 67.9 ⚠️ | 2.4 ✅ | 7.4 / 3.2 / 3.2 | 17.5 | **11.49** | 1.5× | R1 step3 离群；#36 后走廊旋转回退追加扫描（4.79→11.49s，待按方向缓存优化） | ⚠️ |
| dolly_out | 16.0 | 3.7 ✅ | 6.0 / 2.2 / 4.4 | 16.4 | **3.71** | 4.4× | | ✅ |
| orbit_swing | 14.5 | 12.8 ❌ | 7.8 / 74.1 / 73.9 | 159.5 ⚠️ | **4.83** | 33.0× | R2 export 与 retime 各 ~74s（顺序 IK 离群）；批量预解后消除 | ✅ |
| pan_left | 14.0 | 6.9 ✅ | 5.6 / 2.2 / 2.4 | 13.9 | **3.57** | 3.9× | | ✅ |
| pan_right | 14.0 | 7.4 ✅ | 5.7 / 2.2 / 2.4 | 14.0 | **3.61** | 3.9× | | ✅ |
| static_shot | 14.1 | 5.3 ✅ | 4.7 / 2.2 / 2.4 | 13.0 | **3.38** | 3.8× | | ✅ |
| truck_left | — | — | 7.8 / 2.2 / 47.0 | 60.8 ⚠️ | **4.74** | 12.8× | R1 未测；R2 retime 离群 47s | ✅ |
| truck_right | — | — | 7.6 / 2.2 / 46.4 | 59.9 ⚠️ | **4.57** | 13.1× | 同上 | ✅ |
| zoom_in | 14.1 | 5.3 ✅ | 4.7 / 2.2 / 2.4 | 13.1 | **3.35** | 3.9× | R1/FM 仅 zoom 一项 | ✅ |
| zoom_out | 14.1 | 5.3 ✅ | 4.6 / 2.3 / 2.4 | 13.1 | **3.40** | 3.9× | 同上 | ✅ |

**fast <5s：11/12 ✅**（唯一例外 dolly_in 11.49s——#36 走廊旋转回退代价，属正确性换时长的已知取舍，待缓存优化；其余均值 ~3.9s，R2→fast 提速 3.5×–33×）

R2 三列为 step3 规划 / export / retime 实测值，R2 合计 = 5 个重步（step3/export/retime/view/simview）之和；
R2/fast 列已于 9月25 由 `build_timing_table.py` 在 5090-jarvis 上从原始日志现场解析复核，与 9月24 上报数值逐格一致。

## R1 明细（截图存档 · 12 步各步耗时，秒）

| 运镜 | step3 规划 | export | retime | 其余 | 合计 |
|---|---|---|---|---|---|
| dolly_out | 5.9 | 2.2 | 4.2 | 3.7 | 16.0 |
| pan_left / pan_right | 5.6 | 2.3 | 2.4 | 3.7 | 14.0 |
| orbit_swing | 5.5 | 2.7 | 2.6 | 3.7 | 14.5 |
| static_shot / zoom | 5.7 | 2.2 | 2.4 | 3.8 | 14.1 |
| dolly_in ⚠️ | 59.6 | 2.2 | 2.4 | 3.7 | 67.9（10× 离群） |
| crane_up / crane_down | FAIL（可行高度带拒绝：`vertical arc cannot be compressed`） | | | | |

## 重新生成

在有 move_compare 日志的机器上：

```bash
scripts/move_compare/rerun_fast.sh            # 重跑全部单运镜 fast 链
python3 scripts/move_compare/build_timing_table.py
# → cinema_traj_out/move_compare/TIMING.fast.md（含 R2 各步明细：step3/export/retime）
```

`build_timing_table.py` 从 `*.baseline.plan.log` / `*.fast.plan.log` 现场解析 R2 与 fast 列；
R1 与 FM 为 9月23 矩阵的存档常量。
