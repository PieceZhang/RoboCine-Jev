# Stage 3:Shot-aware 预测式 WBC 与 KC-SVSDF

这是 JVS 差速底盘 + 六关节相机臂的实验性集成。早期各阶段的 Isaac 对比在
[生成的报告](results/stage3_report/REPORT.md)中；当前 combo 的正式验证索引为
[shot_validation.json](trajectories/shot_validation.json)。

当前 combo 的衔接修复及八条实际仿真结果见
[衔接抖动核查与修复](#combo-衔接抖动核查与修复2026-10-02)。

## Combo 筛选记录（2026-10-02，完整轨迹口径）

按 groundtruth KC-WBC 的完整轨迹最大位置误差 ≤5 cm 筛选，仅保留以下四条，
编号从 1 连续排列。重命名没有改变轨迹内容，SHA-256 与真实 ROS/Isaac 验证
记录一致；位置 RMS ≤5 cm、姿态 RMS ≤0.1 rad 和运镜动作验收也均通过。

| 当前文件 | 原文件名 | 位置 RMS | 最大位置误差 | scene Y-up 起点 `(x,z,yaw)` |
|---|---|---:|---:|---|
| [shot_combo1.json](trajectories/shot_combo1.json) | shot_combo_reachable.json | 1.05 cm | 3.55 cm | `(5.0,1.0,0)` |
| [shot_combo2.json](trajectories/shot_combo2.json) | shot_combo1.json | 1.03 cm | 3.96 cm | `(4.0,1.2,0.28)` |
| [shot_combo3.json](trajectories/shot_combo3.json) | shot_combo2.json | 1.56 cm | 4.34 cm | `(3.5,2.0,0.62)` |
| [shot_combo4.json](trajectories/shot_combo4.json) | shot_combo3.json | 0.75 cm | 2.93 cm | `(5.8,0.5,-0.35)` |

旧 combo2 已用最终算法补做真实 Isaac 验证，峰值由旧算法的 14.23 cm 降至
4.34 cm，因此没有按旧报告误删。原 `shot_combo.json` 峰值 65.42 cm，不达标，
已从可运行轨迹目录删除；其内容仅保留在 `test/fixtures/combo_unreachable.json`
作为避障负对照，可用于恢复。当前验证索引为
[shot_validation.json](trajectories/shot_validation.json)。旧索引与原始日志保留在
`results/combo_filter_20261002/` 及原结果目录中，历史章节的旧名称不重新编号。
启动包装脚本默认选择 `shot_combo1.json`；两个 launch 的默认 MoveOut 没有改变。
生成脚本只生成编号 1～4 的保留组合，不再生成被淘汰的组合或 `_reachable` 别名。
该轮 103 项回归测试通过；这些历史筛选结果仍不等于完整物理步约束验收。

## Combo 动作序列与自动衔接（2026-10-02）

每个编号现在有两种输入：`shot_comboN.json` 保留原来的完整轨迹；
`shot_comboN_actions.json` 只保存相同的 Orbit/Pan/MoveOut 动作样本，没有
主体获取、动作间衔接或末端停留的样本，也没有全局执行时间戳。
两种格式均使用原来的 `jvs_kc_wbc_trajfollow.launch.py` 和同一套 KC-WBC，
通过 `trajectory_file` 选择，不需要新增 launch 或执行状态机。

当前状态：四组两种格式已使用冻结的同一控制器版本完成真实 Isaac/ROS 成对验证，
八条均通过 5 cm 动作精度、运镜和连续性检查，成对差异也通过；具体数据和衔接
抖动修复见下文。早期 `paired_v1` / `paired_v2` 的成功与失败日志继续保留在
`results/combo_actions_20261002/`，不替代最终版记录。正式 `shot_validation.json`
已收录按编号成对排列的八条轨迹，保存动作精度、全程命令净空和控制器内容哈希；
原四条完整版索引另存为
`results/combo_actions_20261002/final_resume/historical_full_catalog.json`。
最新的八条复测与成对报告在
[final_resume](results/combo_actions_20261002/final_resume/pair_comparison.json)。
停工前的同版本结果保留在
[combo_transition_20261002](results/combo_transition_20261002/after/pair_comparison.json)。

### 生成与运行

从仓库根目录运行：

```bash
# 仅生成四个动作版；不写入现有完整版
/usr/bin/python3 ros2_ws/jvs_mock/scripts/generate_shot_trajectories.py \
  --validation-combos-only --combo-mode actions

# 同时生成两种格式；省略 --combo-mode 时仍是原来的 full 行为
/usr/bin/python3 ros2_ws/jvs_mock/scripts/generate_shot_trajectories.py \
  --validation-combos-only --combo-mode both

scripts/run_kc_wbc_ros.sh \
  --trajectory-file ros2_ws/jvs_mock/trajectories/shot_combo1_actions.json
```

新版 JSON 格式标识为 `jvs_relative_shot_sequence/v1`。`shots` 内每个动作
包含 `type`、局部 `duration_s`、`samples` 以及原来的目标和运镜参数；
每段 `samples[].time_from_start` 从零开始。文件顶层 `duration_s` 是纯动作
时长之和，不是包含自动衔接的执行时长。全部位姿仍相对于同一个启动相机坐标系，
不是相对于前一动作末端。旧 `jvs_relative_task_space_trajectory/v1` 解释方式不变。
新版支持的是 WBC follower；旧的机械臂单独回放器不负责自动衔接。

### 运行时公式

`CameraTrajectory` 运行时将独立动作编译成执行参考，不向输入 JSON 插入衔接：

- 动作位置和符号连续四元数采用插值五次样条，端点一、二阶导数为零，
  保持原运镜采样位姿。角速度从归一化四元数的解析导数计算。
- 衔接采用 SE(3) 最小 jerk 曲线 `s(u)=10u³−15u⁴+6u⁵`：
  `p=p0+s·(p1−p0)`，`R=Exp(s·Log(R1 R0ᵀ)) R0`。
  相邻动作有位姿间隙时自动连接；端点相同则生成静止窗口供 WBC 准备底盘/机械臂。
- 时长取预览/混合时间、平移和转动速度/加速度下界的最大值。
  quintic 曲线的最大归一化速度为 `15/8`，最大加速度为 `10√3/3`；
  基础限值取执行器上限的保守比例，并计入底盘下一段切向转角（模 π）。
  初始主体获取和末端停留同样由执行器生成。实际时间线、参数和生成的衔接
  写入 `reference.json`，诊断脚本据此重建，避免误把纯动作时长当执行时长。
- 自适应路径时钟采用连续误差能量公式
  `rate*=clip(1/(1+(ep/σp)²+(er/σr)²), rate_min, 1)`；
  `σp/σr` 取精度门槛的一半，5 cm 位置目标对应 `σp=2.5 cm`。
  原公式在 10 cm 以内不减速，会在关节接近上限、误差已达 7 cm 时恢复满速。
  衔接准备速率同样用连续的方向/伸出能量衰减，替代 0.35 rad / 0.35 m 的切换。
  路径不跳点，速率变化仍限制为 0.4/s；动作可更慢执行，非固定时刻跟踪保证。

参考生成不是碰撞可达性保证；硬 CBF、关节/速度/加速度约束和发布前连续 sweep
检查仍覆盖每个命令，包括所有衔接和停留。没有为某条 combo 增加特殊状态分支。

### 精度口径与成对验证

新版 `summary.json` 的 `position_error_*`、`orientation_error_*` 和
`tracking_accuracy_pass` 仅统计各段 `[motion_start_s, motion_end_s]`。
主体构图、单动作 RMS/峰值及动作完成量也按该动作区间计算，不能在后续停留里
追上再算动作已完成。`motion_active` 标记写入原始 CSV。
`transition_*` 与 `full_execution_*` 单列衔接/全程误差供诊断，但不作为精度门槛；
安全和命令连续性验收仍使用全程数据，实测物理约束违例仍如实保留。

两种输入均要求动作位置 RMS/峰值 ≤5 cm、姿态 RMS ≤0.1 rad，并通过运镜完成与
方向、安全净空和命令连续性检查。成对“相似精度”的固定标准为位置 RMS 差 ≤1 cm、
峰值差 ≤2 cm，同时要求轨迹动作样本一致和控制器源码/配置哈希一致。
`analyze_combo_pairs.py` 独立按运行时时间线复核 CSV 的动作掩码和统计量，
不同控制器版本、场景或初始底盘位姿的测试不能作为同一组对照。
未完成、含失败原因或缺少全程净空数据的记录也不得进入正式索引。

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
export OMNI_KIT_ACCEPT_EULA=yes OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/usr/bin/python3 ros2_ws/jvs_mock/scripts/validate_kc_shots.py \
  shot_combo1 shot_combo1_actions shot_combo2 shot_combo2_actions \
  shot_combo3 shot_combo3_actions shot_combo4 shot_combo4_actions \
  --output /tmp/jvs_combo_pair_recheck --mesh-cache-root /tmp/jvs_combo_pair_recheck \
  --domain 213 --criterion both
/usr/bin/python3 ros2_ws/jvs_mock/scripts/analyze_combo_pairs.py \
  /tmp/jvs_combo_pair_recheck --plot

# 仅在全部通过且源码哈希仍一致时更新正式索引
/usr/bin/python3 ros2_ws/jvs_mock/scripts/analyze_combo_pairs.py \
  /tmp/jvs_combo_pair_recheck --write-trajectory-index
```

网格缓存由场景与出生位姿指纹核验，同组动作版可复用完整版缓存；
不同起点仍使用独立网格。请使用新的结果目录，旧成功/失败记录不覆盖。

### 续作复测与正式索引

其他修改停止后，重新构建 `jvs_mock`，从新启动的 Isaac/PhysX 场景逐条复跑
四组完整版和动作版。八次均完成，控制器源码、配置和轨迹哈希未在运行中改变；
每组的场景和初始位姿相同。精度仅统计动作区间，所有衔接仍参与命令净空和
连续性检查。原始 CSV 按运行时动作时间线独立复算，四组成对验收全部通过。

| Combo | 完整版动作 RMS / 峰值（cm） | 动作版动作 RMS / 峰值（cm） | 完整版 / 动作版执行时间（s） |
|---|---:|---:|---:|
| 1 | 0.53 / 1.76 | 0.57 / 1.72 | 86.45 / 89.25 |
| 2 | 0.62 / 1.40 | 0.70 / 1.42 | 90.62 / 97.60 |
| 3 | 1.07 / 2.34 | 1.18 / 2.32 | 110.02 / 122.28 |
| 4 | 0.62 / 2.33 | 0.57 / 2.48 | 84.13 / 89.30 |

最大动作段位置峰值 2.483 cm、RMS 1.176 cm。成对峰值差最多 1.531 mm，
RMS 差最多 1.056 mm；全程执行命令的扫掠净空下界最小 12.663 cm。
路径时钟会连续减速，不跳过运镜点；上表不是固定时间跟踪保证。

[正式索引](trajectories/shot_validation.json) 已更新为这八条复测记录，
按 `combo1` 完整版/动作版、`combo2` 完整版/动作版的顺序排列。
验收脚本拒绝未完成、失败、精度口径错误、缺少全程净空或场景/起点不一致的对照；
目录回归还核对当前源码哈希，防止把改算法之前的成功记录当作当前验证。
147 项回归测试通过，两种格式重新生成后八个文件均逐字节一致，构建通过。

原始记录与[成对报告](results/combo_actions_20261002/final_resume/pair_comparison.json)、
[精度对比图](results/combo_actions_20261002/final_resume/pair_accuracy.png)
保留在 `results/combo_actions_20261002/final_resume/`。
本轮未新增控制状态或放宽执行约束，沿用已验证的共同 QP 和衔接修复。
实测物理约束偏差仍如实记录，`stage1_pass=false`；这些跟踪和命令验收结果
不等同于完整物理步审计，也不构成实机安全认证。本轮未推送。

## Combo 衔接抖动核查与修复（2026-10-02）

已检查当前全部四组、两种输入格式，共八个 combo 文件及 16 个衔接窗口。
完整版 JSON 已写出衔接处的静止相机位姿样本；动作版在加载时生成衔接。
当前八个文件的相邻动作端点重合，因此每个窗口的相机参考都是固定位置、固定朝向。
离线检查了窗口内 81 个位姿以及两侧速度、加速度，参考本身连续。
底盘可达位置、前进/后退方向、转向和机械臂配合始终由运行时 KC-WBC/MPC 计算。

### 已确认的原因与修复

- 停车引导原来在 7 cm 距离处切换，并在每个周期重选前进/后退；后续静止引导
  又在 2.5 cm 处切换朝向。反馈噪声会反复跨越边界，产生速度和目标朝向跳变。
  现改为连续的小偏移恢复，每个窗口只提交一次停车方向与到达阶段，衔接两端和
  下一动作开始处均平滑混合；到达后不因噪声重新进入停车阶段。
- 中立姿态作为每次 IK 的固定初值，会在底盘接近半圈转向时切换姿态分支。
  原始实测相邻状态只转动约 1.7°，软底盘目标却变化约 9.8 cm。静止/低速时
  现在从实测关节姿态求解，随后随相机速度连续恢复移动时的姿态偏好。
- 控制器原来允许 joint1/joint6 都到 ±90°，URDF 和位置接口实际只允许 ±80°/±40°。
  限位不一致使位置命令积分到物理止挡之外，机械臂和底盘持续补偿。
  `jvs_kinematics.py` 已与真实限位对齐，新增回归逐关节核对 URDF 和接口。
- 后台预测可能用旧状态覆盖执行阶段已选的分支。现在仅执行器提交这些分支，
  worker 返回结果保留实际方向、停车阶段和进入窗口时的参考。
  方向评分同时计入可实现的相机高度和朝向误差，避免只按舒适程度选择错误方向。
- 停留窗口提高固定相机任务的相对权重、降低软转向速率，并为停车保留准备时间。
  不同位置/朝向之间的移动衔接继续使用正常轨迹跟踪。
  方向姿态评分在后台预计算；相同位姿、高度、朝向和关节初值的 IK 结果复用，
  避免衔接时重复计算导致控制周期超时。

发布命令仍由原来的硬约束 QP 和连续 sweep 检查产生。物理关节限位已收紧，
碰撞净空、速度、加速度和残差判定没有放宽。

### 最终验证

使用冻结的同一控制器版本，逐条启动真实 Isaac Sim/PhysX 和 ROS 的
`jvs_kc_wbc_trajfollow.launch.py`，按文件指定起点运行。八条全部完成，
通过动作精度、运镜完成量/方向、命令连续性和命令净空检查；四组成对精度也通过。
本次使用 launch 默认的 `mesh_source=groundtruth`，关闭未参与控制的 RTX LiDAR。

| Combo | 完整版动作 RMS / 峰值（cm） | 动作版动作 RMS / 峰值（cm） | 完整路径峰值：完整 / 动作（cm） |
|---|---:|---:|---:|
| 1 | 0.48 / 1.70 | 0.53 / 1.72 | 1.70 / 1.72 |
| 2 | 0.53 / 1.61 | 0.70 / 1.65 | 1.61 / 1.65 |
| 3 | 1.04 / 2.24 | 1.11 / 2.64 | 2.24 / 2.64 |
| 4 | 0.62 / 2.34 | 0.57 / 2.49 | 2.34 / 2.49 |

表中完整路径峰值包含衔接（`phase=tracking`），启动准备和末端停车样本另保存在 CSV。

下表按当前编号比较原来四条完整版的各处衔接。完整相机朝向包括 roll，
不仅统计相机视线。高频 RMS 按物理仿真时间重采样至 50 Hz，扣除 σ=0.2 s 的
Gaussian 趋势，并排除窗口两端各 0.4 s；慢速漂移和衔接峰值另保存在 JSON 中。

| 衔接 | 底盘目标朝向最大单步变化（°） | 相机位置高频 RMS（mm） | 相机朝向高频 RMS（°） |
|---|---:|---:|---:|
| combo1 orbit -> pan | 6.16 → 0.02 | 0.185 → 0.038 | 0.046 → 0.004 |
| combo1 pan -> moveout | 146.70 → 6.52 | 0.882 → 0.340 | 0.545 → 0.041 |
| combo2 orbit -> pan | 3.73 → 0.09 | 0.140 → 0.046 | 0.141 → 0.006 |
| combo2 pan -> moveout | 22.10 → 5.27 | 1.024 → 0.244 | 0.165 → 0.050 |
| combo3 pan -> moveout | 51.45 → 6.15 | 0.661 → 0.242 | 0.497 → 0.035 |
| combo3 moveout -> orbit | 178.94 → 7.72 | 1.502 → 0.373 | 0.670 → 0.065 |
| combo4 moveout -> orbit | 126.71 → 7.24 | 1.291 → 0.293 | 0.271 → 0.065 |
| combo4 orbit -> pan | 1.66 → 0.27 | 0.172 → 0.253 | 0.108 → 0.014 |

八个最终文件的全部窗口中，都未记录到超过 15° 的底盘目标朝向单步跳变或
前进/后退分支反复切换。相机仍有反馈与物理仿真误差，不将其表述为绝对零抖动。
更充分的底盘准备会拉长实际衔接时间：例如 combo1 的 Pan→MoveOut 从约 9.45 s
变为约 18.50 s，原来的四秒路径窗口保持不变。

139 项回归通过，包含 24 项衔接专项测试。五项根因测试在任务开始时的源码上
均能复现失败，在修复后全部通过。当前源码和八次实际验证的内容哈希一致。
这些结果来自 ROS 控制周期采样；完整物理步的关节加速度等审计仍需独立执行，
实测物理约束偏差保留在原始统计中。本轮没有做实测 LiDAR 模式的八条复测。

原始 CSV、运行时参考、前后源码、负回归结果、最终统计与对比图保存在
[combo_transition_20261002](results/combo_transition_20261002/comparison.json)，
[衔接对比图](results/combo_transition_20261002/junction_comparison.png) 和
[成对验证](results/combo_transition_20261002/after/pair_comparison.json)。
combo1 在 IK 结果复用前的一次控制周期超时失败及当时源码，也保留在
`results/combo_transition_20261002/trials/integrated_v8/`。
服务恢复发生在 combo3 动作版运行中，该控制器仍完成轨迹；它的完成统计由
`launch.log` 中首个 `WBC_RESULT` 恢复，内容哈希重新核验，元数据注明恢复来源。
恢复后只补跑尚未启动的 combo4 动作版。其他开发中的中止实验没有当作最终通过记录。

从仓库根目录复核保存的衔接记录：

```bash
PYTHONPATH=ros2_ws/jvs_mock /usr/bin/python3 \
  ros2_ws/jvs_mock/scripts/analyze_transition_jitter.py \
  ros2_ws/jvs_mock/results/combo_transition_20261002/after/shot_combo{1,2,3,4}{,_actions} \
  --output /tmp/combo_transition_jitter.json
```

## 各阶段对比

| | Stage 1 `jvs_wbc_isaac` | Stage 2 `jvs_shot_wbc_isaac` | Stage 3 KC-WBC |
|---|---|---|---|
| 控制器 | `WholeBodyQP` | `AsyncShotPredictiveQP` | `AsyncObstaclePredictiveQP`(KC-SVSDF)|
| 运镜感知 | ✗ | ✓ Orbit/Pan/MoveOut | ✓ |
| 预测时域 | ✗(瞬时 QP)| 8 × 0.4 s | 8 × 0.4 s + A* 引导 |
| 障碍规避 | ✗ | ✗ | ✓ KC-SVSDF + 物理障碍场景 |
| 默认轨迹 | `wbc_coordinated.json` | `shot_pan.json` | `shot_moveout.json` |
| 额外配置 | 基础 yaml | + shot yaml | + shot yaml + 障碍场景 |

Stage 1/2 的 launch 仍是一体化入口。Stage 3 拆成
`jvs_kc_wbc_isaac.launch.py` 与 `jvs_kc_wbc_trajfollow.launch.py` 两个独立
进程组：前者只运行 Isaac，后者让 WBC 独占 `/isaac_joint_commands`。Stage 1
解决“底盘和手臂怎么协同跟踪一条相机轨迹”，Stage 2 加上“按拍摄手法（运镜）
分配运动”，Stage 3 再加上“绕开真实障碍物”。

## Launch 职责拆分

| Launch | 职责 | 主要参数 | 明确不负责 |
|---|---|---|---|
| `jvs_kc_wbc_isaac.launch.py` | 启动 Isaac、加载 simview 场景、生成机器人状态与传感器话题 | `stage_view`、`headless`、`x/z/yaw/spawn_height`、`enable_lidar` | 不读取轨迹，不提取 KC 网格，不启动控制器，不发送控制指令 |
| `jvs_kc_wbc_trajfollow.launch.py` | 重置并校验机器人，准备/复用碰撞网格，读取轨迹，运行 KC-WBC，发布 `/isaac_joint_commands` | `trajectory_file`、`controller_mode`、`mesh_source`、`mesh_cache_file`、`output_directory` | 不启动或关闭 Isaac |

两个 launch 必须处于相同的 ROS domain，并使用相同的 `stage_view`、`x`、`z`、
`yaw` 和 `spawn_height`；否则仿真场景与控制器碰撞网格不在同一坐标系。二者的
生命周期彼此独立。正常停止时建议先停止轨迹跟随 launch，使控制器先发布停止
指令，再停止 Isaac。

重新运行前必须先结束上一个 trajfollow 进程，避免旧控制器继续占用
`/isaac_joint_commands` 与 reset client 竞争。

### 每次 trajfollow 启动前的重置

每次运行 `jvs_kc_wbc_trajfollow.launch.py` 都执行以下顺序，不能跳过重置直接启动
控制器：

1. `isaac_reset_client` 持续向 `/isaac_joint_commands` 发布零速、机械臂零位保持
   命令，并等待 Isaac 中的 `/jvs/reset_robot` 服务。
2. Isaac 将机器人 root pose、全部可动关节位置、root 速度和全部关节速度恢复到
   启动状态。
3. reset client 连续收到若干组满足容差的 `/joint_states` 与
   `/base_controller/odom` 后，在终端打印 `ISAAC_RESET_COMPLETE`（Isaac 声明的
   状态）和 `ISAAC_START_STATE`（ROS 反馈观测状态）。
4. reset client 正常退出后，launch 才准备碰撞网格并启动
   `whole_body_controller`；WBC 自身还有默认 2 秒 settle 阶段，然后才建立相对
   相机轨迹并开始运动。

如果 Isaac 未运行、reset 服务超时、物理状态设置失败或反馈没有收敛，trajfollow
launch 会关闭，KC-WBC 不会启动。重置只恢复机器人状态，不回拨 `/clock`，因此
每次新启动的 WBC 使用当前连续的仿真时间。

服务端在同一仿真步内回读并严格校验 root pose、全部关节位置、root 速度和全部
关节速度。随后 ROS 反馈还必须连续 5 组满足默认容差：机械臂位置 `0.005 rad`、
受控关节速度 `0.05 rad/s`、底盘线/角速度 `0.005`（对应单位分别为 m/s 与
rad/s）。由于重力接触和 PhysX 落稳，终端打印的观测值可能有小幅非零偏差；打印
值是控制器实际开始前的状态。

默认启动状态如下：

| 状态项 | 默认值 |
|---|---|
| simview Y-up 场景中的底盘 | `x=5.0 m`、`z=1.0 m`、`yaw=0 rad`、离地高度 `0.002 m` |
| Isaac Z-up 原生底盘位姿 | 位置 `[5.0, -1.0, 0.002] m`，yaw `-pi/2 rad`，roll/pitch 为零 |
| 六个机械臂关节 | `q1..q6 = 0 rad` |
| 左右驱动轮及六个被动 caster DOF | 关节位置 `0` |
| root、全部关节速度 | `0` |
| 仿真时钟 | 保持连续，不重置 |

这里的底盘场景位姿由 Isaac launch 的 `x/z/yaw/spawn_height` 决定；表中是默认值。
trajfollow 中同名参数仅用于构建同坐标系的碰撞网格，不能用于把 Isaac 中已加载的
机器人改到另一出生点。若要改变出生点，必须给两个 launch 传入完全相同的值。

## 模型与优化

状态仍为 `[x,y,yaw,q1..q6]`,输入仍为 `[v,omega,dq1..dq6]`。
`ObstaclePredictiveQP` 扩展了 Stage 2 的 8 × 0.4 s 预测时域。相机
位置/姿态、连续混合的 Orbit/Pan/MoveOut 偏好、heading、posture、命令平滑与
碰撞代价共享同一组 64 个全身控制变量。保留 forward/reverse MoveOut 候选求解。

两个避障变体共享平面 A* homotopy guide、受约束的 successive linearization
(每次更新两次)、trust bounds 和同一个执行 safety QP。guide 使用保守的平面
底盘包络,只是软性位置参考;其边从不作为命令下发给底盘。随后的全身优化器
考虑全部机械臂连杆。该 guide 是必要的:停止底盘的局部 nonholonomic
linearization 自身无法发现侧向位移。不声称完整复现 MINCO/前端搜索。

相机跟踪是软代价。其在障碍附近权重降低,preview 参考与间隙约束优先。快速
50 Hz QP 保留原有 joint position/braking、velocity/acceleration 与
wheel speed/acceleration 约束;通过 `xdot = S(yaw)u` 消除 nonholonomic 运动。
一个共享的间隙导数约束使用实测构型,并留 6 mm 伺服余量。返回的命令还必须
通过连续 constant-control sweep 证书才能发布。不可行或不可证的命令触发
fault;不会被裁剪或静默接受。planner 失败会请求约束制动并记录日志。
安全停机不计为成功完成轨迹。

ROS 节点在派生的 worker 中运行同样的 preview。晚于 0.6 s 到达的 plan 被拒绝;
已接受的 plan 在其替代者计算期间可以老化。本地执行 QP 仍检查当前间隙。
本版本不支持动态障碍地图与移动行人。

## 几何与论文对应

参考:[Li et al., arXiv:2608.07005, sections VI-A–C](https://arxiv.org/html/2608.07005v1)。
实现遵循 minimum-over-time 度量、interior ball-radius 构造、上游
accumulator/filter 与 clearance penalty,适配到现有的 velocity MPC。不实现
论文的整个规划栈。

* `kc_svsdf.py`:解析式 box/capped-cylinder/frustum SDF、link transforms 与
  完整的九态距离 Jacobian。覆盖底盘、驱动轮、passive caster 组件、pedestal、
  body RGBD camera/mount、neck、六条机械臂连杆与运动相机。仅视觉的 lidar 无
  URDF collision shape。
* 机械臂圆柱与 box 匹配 URDF collision 尺寸与旋转。光滑底盘 frustum 外接其
  多面体 STL;caster 轮用包络球覆盖所有被动转向姿态。这是保守碰撞模型,
  不是精确的任意网格几何。
* 环境是静态 sphere 的显式并集,link-to-sphere 间隙为精确的外部距离
  `SDF_link(center) - radius`。地面支撑接触被排除。自碰撞不在本 Stage 3
  环境避障测试范围内。
* `sweep()` 评估连续、精确的 constant-control diff-drive flow。其下界扣除
  全局表面速度 Lipschitz 误差。若采样触及工作量上限,则保留更大的误差值,
  而不是声称精确。
* 预测碰撞项在每个区间搜索三个点,并按相应的时间界膨胀。interior ball 收缩
  使用 70 个球面方向加 consistency-boundary 方向,带 safeguarded 迭代。
  KC 为每个障碍累积上游关节方向,并用论文的两上游关节符号规则过滤下一个
  连杆的候选。Per-link SVSDF 仅关闭该过滤器;仍使用正确的全身 Jacobian。
* 安全包络将本体按障碍半径加预防裕度膨胀,使 KC 的 interior 规则能在物理
  穿透前生效。这是对论文的显式预防性扩展。优化使用 chain-consistent 的
  escape gradient;报告的物理间隙绝不用过滤后的 KC 距离替代真实几何间隙。
* 目标函数对 cubic clearance hinge 使用局部二次近似。有限方向采样与近似
  梯度不是精确的符号或全局收敛 KC-SVSDF 求解器。若在迭代预算内找不到有效的
  过滤 interior 候选,该项保留普通几何梯度。即使优化器报告收敛,仍需 safety
  QP 与独立审计。

## 连续审计与指标

`collision_audit.py` 在每个 1 ms physics step 读取实际的 Isaac 刚体位姿,
独立于控制器 FK,并计算全部本体/障碍距离。相邻位姿之间,它用平移位移加
形状半径 × 旋转角限定完整刚体平移/SLERP sweep。其报告的
`continuous_lower_m` 包含该界。这认证的是记录的 PhysX 位姿的分段刚体
插值,不是真实世界接触保证。physics 审计与相机刚体跟踪测量保持分离。

原始输出:

* `telemetry.csv`:相机目标/实际位姿、位置/姿态误差、base v/omega、全部关节
  速度、shot/phase/next-shot、间隙、碰撞项、preview/求解耗时、硬约束残差与
  planner 失败计数。
* `collision.csv.gz`:每本体间隙、瞬时最小值与 1 kHz 连续下界。
* `physics.csv.gz`:1 kHz 未过滤的实际轮/底盘/关节约束违规,含接触/伺服
  瞬态。
* `summary.json`:完成/失败、跟踪统计、两项独立审计与控制器源码哈希。失败
  运行及其观测时长保留在对比中;安全前缀绝不晋升为已完成的安全运行。

固定评估门槛:全路径观测下界 >=25 mm;完成完整轨迹;最后一秒相机 RMS
<=20 mm 且 <=0.035 rad;命令 residual <=1e-6;归一化命令 step <=0.05;以及
每个 physics step 均满足继承的物理限制。避障期间的临时相机误差允许并如实
报告。collision rate 指**记录的 physics 状态**中保守模型间隙为负(包络
相交,不是独立 PhysX 接触计数)的比例。认证为正的间隙对 URDF 是保守的。
每个 case/method 只有一次确定性试验;这不是统计成功率 benchmark。中止运行
的 rate 只覆盖其观测前缀。

## 复现

在仓库根目录:

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
export OMNI_KIT_ACCEPT_EULA=yes  # explicitly authorized by the user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 ros2_ws/jvs_mock/scripts/generate_obstacle_scenes.py
python3 ros2_ws/jvs_mock/scripts/test_collision_suite.py \
  --output ros2_ws/jvs_mock/results/stage3_reproduction
```

套件以 PGS、physics 1000 Hz、控制 50 Hz 启动真实 Isaac PhysX,三种模式使用
相同的机器人/驱动设置。需要 GPU。这些对比是同步的:求解期间仿真时间暂停。
因此其跟踪结果不能确立实时 ROS 性能;单独的异步 ROS 运行也纳入验收。
使用新的输出目录。场景独立于控制器模式固定两个障碍位姿:一个低底盘障碍与
一个抬高机械臂障碍。MoveOut 使用 `shot_moveout.json`(18 s);组合 case 使用
`shot_combo.json`(Orbit 22 s → Pan 14 s → MoveOut 18 s)。各含 3 s 恢复/制动
测量。Combo 使用与 Stage 2 相同的侧向初始机械臂构型。两个变体均保留未修改的
输入相机轨迹文件。

构建包之后，在第一个终端只启动 Isaac 和场景：

```bash
source /opt/ros/humble/setup.bash
source /home/w84384651/Documents/RoboCine-Yuelin/ros2_ws/install/setup.bash
export OMNI_KIT_ACCEPT_EULA=yes
export ROS_DOMAIN_ID=176
ros2 launch jvs_mock jvs_kc_wbc_isaac.launch.py \
  stage_view:=/home/w84384651/Documents/RoboCine-Yuelin/cinema_traj_out/kujiale_0021_kid_combo/simview/simview.usda
```

两个 KC-WBC launch 默认显式设置 ROS domain 176，不继承终端中可能残留的
domain 0，防止其他仿真或局域网节点的 `/isaac_joint_commands` 串扰。更换 domain
时，两条命令都增加相同的 `ros_domain_id:=<编号>`。手工运行 `ros2 topic`、RViz
或其他 ROS 节点时，仍需在该终端设置 `export ROS_DOMAIN_ID=176`（或指定的编号）。

在第二个终端启动轨迹读取、KC-WBC 和控制指令发送：

```bash
source /opt/ros/humble/setup.bash
source /home/w84384651/Documents/RoboCine-Yuelin/ros2_ws/install/setup.bash
export OMNI_KIT_ACCEPT_EULA=yes
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export ROS_DOMAIN_ID=176
ros2 launch jvs_mock jvs_kc_wbc_trajfollow.launch.py \
  trajectory_file:=$PWD/jvs_mock/trajectories/shot_combo1.json \
  stage_view:=/home/w84384651/Documents/RoboCine-Yuelin/cinema_traj_out/kujiale_0021_kid_combo/simview/simview.usda \
  mesh_source:=groundtruth
```

第二个命令首次运行会生成
`cinema_traj_out/kujiale_0021_kid_combo/simview/simview.kc_mesh.npz` 和同名
`.ply`/`.metadata.json`；后续运行若 fingerprint 匹配会直接复用缓存。若不希望
把缓存写在场景旁，可显式指定：

```bash
ros2 launch jvs_mock jvs_kc_wbc_trajfollow.launch.py \
  stage_view:=/home/w84384651/Documents/RoboCine-Yuelin/cinema_traj_out/kujiale_0021_kid_combo/simview/simview.usda \
  mesh_source:=groundtruth \
  mesh_cache_file:=/home/w84384651/Documents/RoboCine-Yuelin/ros2_ws/jvs_mock/results/kujiale_groundtruth/scene_mesh.npz \
  trajectory_file:=/home/w84384651/Documents/RoboCine-Yuelin/ros2_ws/jvs_mock/trajectories/shot_combo1.json \
  output_directory:=/home/w84384651/Documents/RoboCine-Yuelin/ros2_ws/jvs_mock/results/kujiale_groundtruth/combo
```

需要忽略已有缓存并重新从 USDA 提取时，在上述命令后增加
`force_mesh_rebuild:=true`。

Isaac 已在第一个终端运行时，快速单跑 KC 模式的 combo：

```bash
ros2 launch jvs_mock jvs_kc_wbc_trajfollow.launch.py \
  trajectory_file:=$PWD/jvs_mock/trajectories/shot_combo1.json \
  stage_view:=$PWD/cinema_traj_out/kujiale_0021_kid_combo/simview/simview.usda \
  mesh_source:=groundtruth
```

Isaac launch 默认使用文档中的 simview、`q=0`、开启双 LiDAR，并且始终关闭旧的
command adapter 与 trajectory follower。轨迹跟随 launch 默认使用 MoveOut、
`mesh_source:=groundtruth` 和 collision mode `kc`。设置
`controller_mode:=per_link` 或 `none` 可使用匹配的替代模式。
`/wbc/shot_context` 保留 Stage 2 的 JSON 输入。`mesh_cache_file` 为空时缓存写到
`<stage>.kc_mesh.npz`；缓存 fingerprint 包含 USD 路径、大小、mtime、出生位姿和
网格采样参数。`force_mesh_rebuild:=true` 可强制重新提取。

## 用户提供的 simview 房间

所请求的 `cinema_traj_out/kujiale_0021_kid_combo/simview/simview.usda` 已在
Isaac 中加载并检查,见
[inspection.json](results/stage3_simview/inspection.json)。它由 1,360 个
mesh prim、198 个显式 collision prim 与 3 个动画 prim 组成。仅凭原始计数
不能确立碰撞覆盖,因为 USD schema 可能写在父节点上。现有 simview runner 将
Y-up `(x,y,z)` 映射为 Z-up `(x,-z,y)`,并将它的零朝向映射为该机器人的
`-pi/2`。

当前 groundtruth 地图会遍历 USD 中的 `UsdGeom.Mesh`,应用完整 world transform，
按 simview 的 Y-up → Isaac Z-up 规则及机器人出生位姿变换到 odom。地板和天花板
从避障地图中排除；指定高度和半径内的三角形被保存在 NPZ/PLY，同时以体素外接球
作为 KC-SVSDF 的保守表面采样。机器人每个碰撞体只把最近的 4 个表面 witness 和
全部进入 8 cm 安全激活区的 witness 放入快速 CBF QP，沿扫掠采样选出的局部保守
表面集合仍由连续命令 sweep 检查。LiDAR 传感器由 Isaac launch 的
`enable_lidar:=true`（默认）启用；`mesh_source:=lidar` 接入实时双路点云，
详见 [实时 LiDAR 接口与测试](LIDAR_WBC.md)。

与安全相关的实现细节:执行 QP 连续约束每一对 场景障碍/机器人基元,而不只是
复合底盘 SDF 的当前最小值。没有硬距离激活截止。这避免了最近形状切换或新
启用 barrier 行导致的急刹需求,后者可能与加速度界冲突。对 ROS,收到的 plan
与 warm start 在使用前先推进到其当前龄期。被拒绝的 QP 会写出
`failure_qp.json`,包含状态、上一命令与完整约束矩阵供诊断。数值有效的 MPC
命令仍不是实际伺服/接触加速度的保证。

七个数值契约测试检查本体与基元距离梯度、运动学一致性、一个藏在安全端点之间
的碰撞、时间界、实测底盘倾角的保持,以及 KC 候选过滤器。它们只是数学检查;
全部机器人验收结果来自 Isaac。仿真器的环境与控制器场景使用同一个 JSON。
房间检查单独保存,不计入碰撞成功率。

URDF 几何交叉检查保存在
[urdf_check.json](results/stage3_geometry/urdf_check.json):全部 19 个基元
collision shape 匹配或被保守包络,全部 384 个 STL 三角形顶点在数值容差内
位于光滑底盘 frustum 之内。

## 验收局限

KC 同步最终测量:

| Case | 连续间隙下界 | 位置 RMS | 最后一秒位置 RMS | 完成 |
|---|---:|---:|---:|---|
| MoveOut | 63.18 mm | 22.95 mm | 2.11 mm | 是 |
| Orbit → Pan → MoveOut | 31.19 mm | 21.90 mm | 5.73 mm | 是 |

两者观测到的包络相交均为零,且满足 25 mm 最小值。组合 case 中,base 偏离
空场景 Stage 2 参考最多 0.265 m,机械臂构型在关节向量范数上最多偏离
2.099 rad。这确立了该 case 中 base 与机械臂的联合响应。KC preview 求解
耗时均值 282.4 ms / P95 468.2 ms;它不是 50 Hz 的 MPC 求解。快速 constrained
execution QP 以 50 Hz 单独运行。普通 per-link 组合运行在约 39.4 s 处停止,
当时一个候选未能通过连续命令 sweep 检查;其记录的安全前缀不能确立全路径
可行性。

Stage 3 **未严格通过验收**。同步 Isaac case 能为其测试路径确立间隙、协同
避障与跟踪恢复,但当前集成存在两个独立失败:

1. 命令空间约束在求解器容差内通过,而 1 kHz 物理审计仍观测到实际加速度
  违规(偶发速度违规)。速度级模型未覆盖全部驱动与接触瞬态。这些测量被
  保留,不做平滑也不放宽继承的限制。
2. 异步 ROS MoveOut 测试走完时间线但未恢复相机目标:位置 RMS 为
   133.94 mm,最后跟踪秒为 210.44 mm RMS。被接受的 preview plan 龄期 P95
   达 0.75 s、最大 1.30 s;fast QP P95 为 1.89 ms。plan 龄期对齐修复未能
   消除状态失配或局部规划失败。

剩余工作是:缩小预测执行/延迟差距、使实际执行器动力学满足限制,以及接入 LiDAR
实时建图器。提高容差或将走完时间线计为成功都不能解决这些失败。完整报告见
生成的报告,包括全部三种方法、失败运行、公共窗口对比、solver 耗时与 replay。

## 真实 simview groundtruth mesh 回归（2026-09-21）

首次提取在出生位姿 `(x=5, source-z=1, yaw=0)` 周围 3 m 范围得到 206 个源
mesh、489,690 个三角形及 9,718 个 8 cm surface voxels；再次启动直接报告
`GROUNDTRUTH_MESH_CACHE_HIT`，不再启动 Isaac 解析 USD。最终代码的四条异步
ROS/Isaac 轨迹结果如下：

| 轨迹 | 完成 | 连续 sweep 最小间距 | 位置 RMS | 姿态 RMS | fast QP 平均/最大 |
|---|---:|---:|---:|---:|---:|
| Pan | 是 | 108.3 mm | 28.3 mm | 0.246 rad | 0.76 / 2.54 ms |
| Orbit | 是 | 120.9 mm | 238.2 mm | 0.130 rad | 0.86 / 2.79 ms |
| MoveOut | 是 | 145.5 mm | 232.2 mm | 0.240 rad | 0.93 / 4.57 ms |
| Orbit → Pan → MoveOut | 是 | 104.5 mm | 280.3 mm | 0.276 rad | 0.91 / 2.65 ms |

所有运行的命令侧 joint/wheel/diff-drive 约束残差最大值低于 `7e-15`，但实测
加速度和横向速度仍违反 Stage 1 限制。尤其 MoveOut 的平均 `|command_v|` 为
0.283 m/s，而平均 `|measured_v|` 仅 0.014 m/s；Orbit 实际底盘总行程仅
0.033 m。因而本次结果验证了 USD 网格缓存、KC-WBC 接入、轨迹完成和保守间距，
但没有通过跟踪精度或物理执行约束验收。逐运行数据和日志位于
`results/groundtruth_real_scene_20260921/`。

## shot_combo 平面引导故障修复（2026-09-22）

在 `kujiale_0021_kid_combo/simview/simview.usda`、出生位姿
`x=5, z=1, yaw=0` 和当前 `shot_combo.json` 上复现了
`no planar collision guide found`。逐圆投影会把参考点推到不连通区域，且逐段
固定大小的搜索框可能截断绕行路径。现在使用覆盖全轨迹的自由栅格连通分量，
把参考点投影到起点可达的区域；引导在经过碰撞检查的线段上插值，避免三次插值
切入障碍。该引导仍只是 MPC 的软参考，不是底盘执行命令。

同时修复了以下运行问题：

- 引导构建失败进入现有 preview 故障处理，记录 `preview_failure`，不再直接
  终止快速控制器；快速 QP 的碰撞约束与连续 sweep 检查仍然执行。
- 过期 MPC 结果中的静态引导可以保留；控制序列超过预测时域后停止沿用末端
  速度。当前代之外的结果仍丢弃。
- 网格执行约束提前纳入接近中的表面点，碰撞减速增益与加速度/速度限制匹配。
- 批量计算碰撞梯度，软碰撞代价每个连杆保留最差的两个网格见证点，减少后台
  interior 计算。硬执行约束与 sweep 证书不采用这一软代价缩减。
- 密集约束导致 OSQP 数值收敛不足时，先收紧求解容差，再按需用 SLSQP 求解
  同一个 QP；所有结果仍必须通过原始 `1e-6` 硬约束残差检查。
- 独立 ROS/Isaac 测试脚本把指定 domain 显式传给两个 KC launch。

同场景真实 ROS/Isaac 复测完整运行 54 秒轨迹及 3 秒停止阶段，共 2,671 个控制
采样，结果为 `completed=true, failure=""`。命令 sweep 最小下界为 29.87 mm
（要求 25 mm），QP 最大残差为 `6.67e-16`，MPC 最大残差为 `9.39e-7`，
过期 preview 丢弃次数为 0；11 次快速 QP 使用 SLSQP 数值后备。

**本次修复消除了控制器故障，没有通过完整跟踪/物理验收。** 位置 RMS 为
0.679 m、最大 1.186 m，姿态 RMS 为 0.313 rad；实测加速度、轮加速度及横向
速度仍超限。相机目标几何检查本身没有发现碰撞，也不能据此证明全身轨迹可行。
不修改原轨迹，不放宽验收阈值。该场景的后半段跟踪和执行器/接触动力学仍需
继续处理；`completed` 仅表示跑完时间线。

本地运行记录位于 `results/kc_combo_debug_20260922/`，包括原始结果、复测
`tracking_summary.json`、`telemetry.csv`、`launch.log` 和额外统计
`diagnostics.json`。针对该故障的回归测试位于 `test/test_collision_route.py`。

## shot_combo 主动底盘跟随与机械臂舒适姿态（2026-09-22）

该轨迹的 Pan 实际带有约 2.4 m 平移，原来的原地 Pan 分配压制了底盘运动。
现按实际平移调整分配，加入折肘预备姿态、按目标高度和姿态生成的关节参考、
零空间回中、伸出距离及可操作性软偏好。底盘在同一受约束 QP 中主动跟随，
落后时通过无碰撞路径点恢复，收尾时回到固定可达位置并释放机械臂偏转。
允许绕行到达的舒适底盘目标不会仅因直线受阻而被放弃。

舒适伸出参考默认 0.25 m；障碍投影要求更大伸展时可放宽这一软参考，底盘
滞后本身不构成放宽理由。平面连通性使用保守模型，不能证明真实全身全局
不可达。所有执行器硬约束、25 mm 最小间隙及验收阈值保留。启动新增 3 秒
预备阶段，保持捕获的初始相机目标，原始 54 秒轨迹及其参考系不变。

此前固定时序版本的同场景真实 ROS/Isaac 测量如下；末帧指 tracking 阶段的最后采样。

| 指标 | 上一轮故障修复后基线 | 本次 |
|---|---:|---:|
| 位置 RMS | 0.6792 m | 0.1848 m |
| 姿态 RMS | 0.3132 rad | 0.0599 rad |
| 末帧位置误差 | 1.1852 m | 0.0410 m |
| 末帧机械臂水平伸出 | 0.5007 m | 0.1854 m |
| 末帧最小关节余量 | 0.0080 rad | 0.5475 rad |
| 底盘实际总路程 | 2.284 m | 5.532 m |

最终运行完成，无控制器故障；命令 sweep 最小下界 54.80 mm，命令侧约束
通过。71 项回归测试通过，包构建成功。**全段精度和物理验收仍未通过**：
床边绕行仍有较大误差，控制回调 P95 为 33.09 ms，实测加速度与侧滑超限。
新增姿态目标使 Orbit 位置 RMS 从 4.3 mm 增至 27.4 mm，改善主要来自后两段。
按时间间隔加权的位置 RMS 为 0.6580 → 0.2179 m；异步运行间仍有差异。

完整原因分析、每段统计、先前复测、验证边界及对比图见
[舒适姿态复测报告](results/kc_combo_comfort_20260922/README.md)。
原始遥测、最终源码哈希和统计脚本也保存在该目录。

## 以运镜动作和衔接为优先的 Kid 场景执行（2026-09-22）

本轮按用户要求采用位置 RMS `0.1 m`、姿态 RMS `0.1 rad`，不再把精确的固定
时刻跟踪作为唯一目标。保留原始几何路径，通过平滑的路径时钟放慢落后段；相机
前馈、底盘前馈及 MPC 的未来参考使用同一时间尺度。MPC 控制序列的年龄及失效
判断仍使用真实仿真时间。JSON 中 `motion_end_s` 到下一段 `motion_start_s` 的
静止窗口用于底盘准备下一方向、移动到可达的舒适位置，并释放机械臂偏转。

主体锁定使用实际相机位置瞄准 kid；遥测的 `orientation_error` 仍计算相对于原
计划姿态的误差。原始组合的 Pan 末端靠近床，理想底盘位置落入家具区域。对这种
投影后才可行的末段 MoveOut，选择固定可达出口并限制行进速度，避免跟随床边
不断变化的最近投影点而反复进退；退出过程中继续偏好紧凑机械臂姿态。
当实测底盘进入保守平面包络内，恢复目标会继续沿可达栅格路径前进到第一个转角，
避免把仅相距几厘米、但需要横向移动的最近自由栅格点当作终点而卡住。

新增三个不同起点、不同顺序的目标锁定组合：

| 文件 | scene Y-up 初始 `(x, z, yaw)` | 动作 |
|---|---|---|
| `shot_combo1.json` | `(4.0, 1.2, 0.28)` | Orbit +15° → Pan −20° → MoveOut 0.45 m |
| `shot_combo2.json` | `(3.5, 2.0, 0.62)` | Pan −15° → MoveOut 0.40 m → Orbit +20° |
| `shot_combo3.json` | `(5.8, 0.5, −0.35)` | MoveOut 0.30 m → Orbit +12° → Pan −10° |

角度使用场景 Y 轴旋转符号；这里的 Pan 保持已有的 `target_locked_arc` 定义。
可在 JSON 中直接设置：

```json
"initial_base_pose": {
  "frame": "scene_y_up", "x": 4.0, "z": 1.2, "yaw": 0.28, "height": 0.002
}
```

这个显式字段优先于 launch 的 `x/z/yaw/spawn_height`。修改它后，目标锁定轨迹
自动围绕同一个世界坐标 kid 重新锚定，保留各段转角和 MoveOut 径向距离。
`scene_context.assumed_spawn` 记录原始 samples 的生成坐标，不必手工改写。
普通无世界主体锚点的相对路径仍需自行重新生成。任意新出生位姿仍须检查碰撞，
只有报告列出的三组出生位姿已经验证。

**改变初始位置需要重启场景，两个 launch 使用同一份 JSON。** 不在运行中偷偷
传送机器人，因为 Isaac 的 odom 原点和碰撞网格坐标都依赖场景启动位姿。
trajfollow 会核对 reset 服务报告的出生位姿；不匹配时明确报错，不启动 WBC。

在 `ros2_ws` 下执行，两个终端均先 `source /opt/ros/humble/setup.bash` 和
`source install/setup.bash`：

```bash
# 终端一：仅启动场景；没有第二个控制命令源
ros2 launch jvs_mock jvs_kc_wbc_isaac.launch.py \
  trajectory_file:=$PWD/jvs_mock/trajectories/shot_combo1.json

# 终端二：reset、核对出生位姿、建立匹配的 groundtruth 网格、运行 WBC
ros2 launch jvs_mock jvs_kc_wbc_trajfollow.launch.py \
  trajectory_file:=$PWD/jvs_mock/trajectories/shot_combo1.json \
  mesh_source:=groundtruth output_directory:=/tmp/jvs_combo1
```

默认 `stage_view` 即本次 Kid 场景；两个 launch 的 `ros_domain_id` 必须相同。
生成三份组合而不覆盖旧轨迹：
`python3 jvs_mock/scripts/generate_shot_trajectories.py --validation-combos-only`。
串行真实仿真验证可使用 `scripts/validate_kc_shots.py`，逐次保存遥测、reset 声明、
原始结果和源码/轨迹哈希，并只关闭脚本自身启动的进程。

`completed` 仍只说明运行结束；`tracking_accuracy_pass` 检查新的 RMS 阈值。
`approximate_success` 还检查每段实际转角/径向位移、反向移动累计量、kid 构图、
命令和采样相机连续性以及 25 mm 扫掠净空，不把跑完时间线当作动作成功。
这里采用完成比例 70%～130%、构图 RMS 0.15 rad；反向量容许
`0.02 rad / 0.05 m + 25% 计划动作量`，在显式 `motion_start_s`～`motion_end_s`
动作窗口中以 0.5 秒间隔统计，并扣除每步 2 mrad / 2 mm 反馈噪声。衔接窗口可以
调整位置和方向，但全程仍检查相机及命令连续性。完整物理步验收另列，未通过时
不会修改 `stage1_pass`。

最终结果、不同起点路径图和逐段统计见
[本轮验证报告](results/kc_shots_20260922/README.md)；轨迹旁的
`trajectories/shot_validation.json` 保存可版本管理的验证索引。

## Combo 最大位置误差 5 cm 优化（2026-10-01）

本轮采用更严格的目标：完整运镜的**最大相机位置误差 ≤0.05 m**，而非仅 RMS
达标。`shot_whole_body_controller.yaml` 同时设置位置 RMS/峰值门槛为 0.05 m，
姿态 RMS 门槛仍为 0.1 rad。本节实验沿用 2026-10-01 的旧名称；原失败轨迹已于
2026-10-02 移为测试夹具，当时的 `shot_combo_reachable.json` 现为
[`shot_combo1.json`](trajectories/shot_combo1.json)。下方运行命令已改为当前编号。
Launch 默认轨迹仍为 MoveOut。

### 诊断：几何可达性与控制偏差都有贡献

在相同 Kid USD、默认出生位姿和真实 ROS/Isaac 下，原控制器的原 combo 位置 RMS
为 23.06 cm，峰值 40.62 cm。命令与实测底盘速度一致，本次不是历史上的速度
桥接故障。主要误差集中在床边 Pan 末段和 MoveOut 前段。

`scripts/diagnose_combo.py` 对全身九态构型做多初值约束 IK，并独立核验全地图
间隙；局部 IK 失败本身不作为全局不可达证明。另用底盘内部圆柱、保留所有不确定
栅格的自由空间超集、去除姿态决定的腕端偏移，以及剩余连杆长度上界，计算必要
可达性下界。在原轨迹 34/36 s，精确目标姿态下位置误差至少为 7.88 cm；即使
允许 0.1 rad 姿态偏差，仍至少为 5.81 cm。

该下界仅针对**直立底盘、当前保守 surfel 球模型和 25 mm 净空**；不是任意倾斜
底盘或原始三角网格的全局物理不可达证明。额外的实际 USD 顶点子集诊断去掉了
voxel 膨胀，仍未找到床边可行 IK，但此乐观子集不是完整碰撞模型，也不足以证明
真实网格不可达。保留当前安全模型时，单纯优化控制器不能满足这些位姿的 5 cm
位置精度与近似目标姿态要求。

新增轨迹仍从 `(x=5, z=1, yaw=0, height=0.002)` 出生，锁定同一个 kid，保留
Orbit +20° / 22 s 和 MoveOut 0.7 m / 18 s，将床边 Pan 从 −55° / 14 s 改为
−30° / 18 s；总名义时长由 54 s 变为 58 s。沿用 quintic 时间曲线和已有静止
衔接窗口。每秒及末端的 59 个采样都有满足姿态、关节限制和全地图净空的静态 IK
见证；这不等于动态路径证明，最终仍以真实 Isaac 执行验证。

### 公式修改：执行精度与预测绕行分开

仅修改通用数值/代价公式，没有新增 shot 专用状态机或特殊轨迹分支：

- `kc_svsdf.py` 将 sphere、box、capped-cylinder、frustum 的六点有限差分法线
  替换为解析法线/有界次梯度，仍使用完整九态距离 Jacobian
  `∂d/∂x = −n_worldᵀ J_point`。距离和碰撞模型不变，所有邻近 primitive 的硬
  约束与发布前 sweep 检查保留。522 个局部 surfel 的梯度查询均值从 7.75 ms
  降至 3.30 ms；单测用独立有限差分核验解析结果及非光滑边界的有限性。
- 快速执行 QP 的相机任务权重为
  `tracking_weight × diag(400,400,400,180,180,180)`，默认系数 10。相机精度
  优先于软底盘引导/姿态偏好，速度、加速度、关节限位与避障仍是原来的硬约束。
  原系数下底盘引导的 1200 权重能在可达路径上留下约 9.5 cm 峰值偏差。
- MPC 保留原来的 `diag(1800,1800,1800,600,600,600) × h × tracking_scale`。
  中间实验曾同时提高 MPC 和执行层权重，虽在新 combo 上通过，但 combo3
  峰值恶化到 54.89 cm。系数为 1 的对照峰值为 11.60 cm，确认这是预测分配
  回归。最终只提高执行层权重，保留预测层绕行与姿态准备的权衡；对应单测要求
  系数 1/10 的 MPC 计划一致。失败实验全部保留，未混入最终重复验证结论。

已有自适应路径时钟、主体锁定和衔接策略没有修改；新轨迹真实执行约 67 s，不能
把 58 s 名义时间当作固定时刻跟踪结果。误差使用反馈 FK 相机位姿与原目标计算，
不删除峰值、不做误差平滑，另报告 P95 和按真实控制间隔加权的 RMS；统计脚本
还核验包括准备/停止阶段在内的全记录峰值。

### 最终版真实 ROS/Isaac 结果

以下三次新 combo 使用完全相同的最终算法与配置；记录的源码/配置哈希均与当前
文件一致。三次都完整执行、通过运镜动作/衔接验收且全记录峰值小于 5 cm。

| 新 combo 最终版运行 | 位置 RMS | 位置峰值 | 位置 P95 | 时间加权 RMS | 最小命令 sweep 下界 |
|---|---:|---:|---:|---:|---:|
| `separate_execution_priority` | 1.047 cm | 3.449 cm | 2.183 cm | 1.183 cm | 8.328 cm |
| `final_v2_verification` | 1.045 cm | 3.185 cm | 2.118 cm | 1.186 cm | 8.738 cm |
| `final_v2_repeat` | 1.048 cm | 3.552 cm | 2.191 cm | 1.197 cm | 8.499 cm |

原控制器仅换新轨迹时 RMS/峰值为 2.602/9.505 cm，所以仅换轨迹不足以达到
峰值目标。最终执行公式在其他已可达组合上也复测：combo1 RMS/峰值为
1.026/3.958 cm；combo3 两次 RMS/峰值为 0.740/2.925 cm 和 0.748/2.933 cm。
未改动这些轨迹，保持不同出生位姿和运镜顺序。它们同样通过运镜动作和精度门槛。
这验证了跨这几条路径的迁移，不声称任意场景的成功保证。

原始不可满足目标的 combo 在最终版下 RMS/峰值为 33.109/65.417 cm，精度和
运镜验收仍失败；不能将最终算法对新轨迹的结果表述为原 combo 已达标。测试
脚本对含这个负对照的批次返回退出码 1 是预期失败，而非忽略失败案例。

最终新 combo 姿态 RMS 为 0.024～0.025 rad，控制回调 P95 为 20.8～21.1 ms，
真实执行时长 67.3～67.4 s。25 mm 净空和所有命令侧硬约束保留。回归测试
`pytest -q ros2_ws/jvs_mock/test tests/test_jvs_kc_launch_domain.py` 共 98 项通过；
`colcon build --packages-select jvs_mock --symlink-install` 成功。
完整 16 次运行（包括中间失败和旧配置）见
[`comparison.json`](results/combo_optimization_20261001/comparison.json)，
误差/间距对比见
[`tracking_comparison.png`](results/combo_optimization_20261001/tracking_comparison.png)。
重建统计与图表：
`/usr/bin/python3 ros2_ws/jvs_mock/scripts/analyze_combo_optimization.py ros2_ws/jvs_mock/results/combo_optimization_20261001`。

### 复现与数据

两个终端均在 `ros2_ws` 下先执行：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export OMNI_KIT_ACCEPT_EULA=yes OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

```bash
# 终端一：场景与 follower 必须使用同一轨迹、出生位姿和 domain
ros2 launch jvs_mock jvs_kc_wbc_isaac.launch.py \
  headless:=true ros_domain_id:=213 \
  trajectory_file:=$PWD/jvs_mock/trajectories/shot_combo1.json

# 终端二：保持既有 reset 与出生位姿核验
ros2 launch jvs_mock jvs_kc_wbc_trajfollow.launch.py \
  ros_domain_id:=213 mesh_source:=groundtruth \
  trajectory_file:=$PWD/jvs_mock/trajectories/shot_combo1.json \
  output_directory:=/tmp/jvs_combo_reachable
```

从仓库根目录可串行重跑：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
export OMNI_KIT_ACCEPT_EULA=yes OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/usr/bin/python3 ros2_ws/jvs_mock/scripts/validate_kc_shots.py \
  shot_combo1 shot_combo2 shot_combo3 shot_combo4 \
  --output /tmp/jvs_combo_5cm_recheck --domain 213 --criterion both
```

每次使用新的输出目录；脚本仅关闭自身启动的进程。生成新轨迹而不覆盖原轨迹：
`/usr/bin/python3 ros2_ws/jvs_mock/scripts/generate_shot_trajectories.py --reachable-combo-only`。
本地全部成功/失败实验、原始 CSV、日志及源码/配置/轨迹哈希保存在
`results/combo_optimization_20261001/`（按仓库规则忽略的运行产物）。必要下界见
[`original_necessary_reach_bound.json`](results/combo_optimization_20261001/original_necessary_reach_bound.json)，
新轨迹静态见证见
[`reachable_all_seconds_ik.json`](results/combo_optimization_20261001/reachable_all_seconds_ik.json)。

验收边界：本轮是异步 ROS 控制频率的 Isaac/PhysX 跟踪验证，没有新做 1 kHz
physics-step 独立审计。实测加速度/横向速度超限仍原样记录，`sampled_checks_pass`
和 `stage1_pass` 未被改成通过。5 cm 跟踪达标不等于全部物理执行约束验收。
