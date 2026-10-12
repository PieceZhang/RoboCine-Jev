## 背景回答（你的问题："新场景规划器的输入到底是什么"）

探查报告结论：你的三个输入方向正确，但现实是"三个半"：
1. **地图点云 .pcd** ✅ 真输入——extract 阶段唯一输入（--map，.pcd/.ply/.npz）→ 自动产出 rooms/structures/objects/doors 草稿；但草稿是无语义标签（object_01），需人工精修（InteriorAgent USD 有语义边车可自动精修）；
2. **极简 yml（moves 一行一镜）** ⬅ 我们正在建——当前卡在 cast/props/audit 还没有"从场景 json 兜底"的代码路径（本计划补上）；
3. **subject pose** ✅ 真输入——scripts/add_subject_track.py（--static "x,z" 或 --waypoints）写进场景 json；
3.5. **环境资产**（隐藏第四项）——Isaac 渲染需要 provenance.env_usd（室内 USD）或点云本身；纯几何管线（盒子舞台）不需要。

cast.json/audit 结果/全部相机坐标/exports/viewer 全是**派生产物**，不是输入。

## 执行内容（已获你批准的方向）

### A. 极简输入落地（三处修改）
1. `tools/scenario_gen.py`（main 函数，~20 行）：
   - yml 有 `moves:` 且无 cast 时 → cast/props/audit 从场景 json 新增的 `"staging"` 字段兜底（yml 显式写的优先）；
   - yml 无 `target` → 用场景 subject_track.subject_id；`plan_id` 缺省 = scene_id。
2. `cinema_traj/scenes/kujiale_0021_kid_combo.json`：加 `"staging"` 字段（cast/props/audit 从当前 yml 原样搬入，附中文注释）——此改动上一轮已完成（脚本执行成功）。
3. `cinema_traj/scenes/config/kujiale_0021_kid_combo.yml`：覆写为你的 9 行精确内容（scene_id/subject/scene + moves 三行，运镜方向"右"按相机右手语义=向相机右手侧绕）。

### B. 荡弧导出单点不可达收尾（悬着的小 blocker）
用新 traj_final 的荡弧段取 t=0.75s 真实位姿+真实切向复探（上次探错文件）：
- 分支锁死 → ik_stream 冷解重试已写好，确认其生效；
- 切向无解 → 调 swing_deg 30→25 或 close_ratio 0.80→0.85 一档；
然后 `--plan-only` 全链（预期：荡弧 100% 可达、遮挡≈0、check_trackable ✅）→ 渲染 POV 核验三镜构图 → 提交推送 PR #23 + 评论（含"新场景三个输入"说明）。

### 不做
- LLM 规划路径不动；PR #26 集成分支保持本地；规划耗时优化另立待办。