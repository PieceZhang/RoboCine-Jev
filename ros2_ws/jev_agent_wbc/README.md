# Jev 相机意图规划与独立执行层

## 单候选构造实验（2026-10-17）

`runner --single-candidate --preplan --mode jev` 在镜头时钟启动前，以实测相机位姿为
起始边界调用官方 TypeSafe Jev。第一次 Choice 限定生成器及位移平面，第二次 Choice 固定唯一
空间种子和分配策略；只生成、优化、检查这一条曲线，失败即结束本次初始规划。
第三次 Choice 绑定通过检查的完整候选。它不做多候选排序，也不会悄悄换方向重试。
原始镜头、终态与 `shot_acceptance/v1` 门槛不变。默认旧入口仍保留供对照。
联合曲线的选定绕行方向不得反转，形状系数的平面法向调整限制在 2.5 cm，
终点仍在原构图误差带内。生成前仅用观测点和机械臂链长必要条件收紧选项，
不预先生成多条候选再交给 Jev 选择。

随机输入默认只扰动机器人冗余构型，再通过 FK 投影保持原相机的起始位置和姿态；
障碍位置及三维速度独立扰动。`--initial-mode independent` 是额外压力测试，
不属于本次保持相机起始位姿的验收范围。

API 连接超时采用相同请求退避重试，记为 `transport_unverified_retry`，不计规划失败；
官方返回类型、选项概率或模型不匹配会单独记录协议错误。密钥只从文件/环境读取。
完整记录、随机扰动范围与验收结果见 `DIRECTED_SEARCH_REPORT.txt`。

```bash
# 创建固定种子的六组扰动输入，并验证单条候选。三个种子均使用独立新目录。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 \
  ros2_ws/jev_agent_wbc/scripts/benchmark_directed_jev.py \
  --output /tmp/jev_single_seed --key-file "$PWD/typesafe_Jev_api_key" \
  --seed 20261017 --startup --diagnose --candidate-seconds 30
# 完整 CPU 闭环。串行测量，实际逐步墙钟计时决定是否满足 20 ms。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 \
  ros2_ws/jev_agent_wbc/scripts/benchmark_official_jev.py \
  --output /tmp/jev_single_closed --key-file "$PWD/typesafe_Jev_api_key" \
  --scenarios /tmp/jev_single_seed/scenarios --single-candidate --preplan \
  --candidate-seconds 30 --proposals 1 --shortlist 1 --pace 1 --workers 1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 \
  ros2_ws/jev_agent_wbc/scripts/analyze_directed_jev.py \
  --runs /tmp/jev_single_closed --independent
```

单候选 CPU 模式使用约 58 个构型检查节点，保留每段中点 FK 和连续碰撞下界；
只有无法判定的自碰撞区间继续细分。联合优化采用 13 个节点和独立的相机穿越探测。
已批准轨迹的在线构型诊断使用向前 2 秒滚动窗口，全程相机/镜头检查与每步安全证书仍保留。
预测、模拟传感器融合和周期几何反馈在独立进程中执行，日志有序异步落盘；
后台进程以较低调度优先级运行，进程预热在镜头时钟之前完成。

`analyze_directed_jev.py` 同时报告完整候选流程和完整主循环计时；阶段二严格标记使用
规划 `<1 s`、**所有**主循环步骤 `<=20 ms`，不以均值或 P99 代替最大值。
API 与预热时间单独记录。实现、失败轮次和最终证据见 [CPU_OPTIMIZATION_REPORT.txt](CPU_OPTIMIZATION_REPORT.txt)。
离线数值性能回放入口 `scripts/benchmark_directed_cpu.py --sources <官方闭环目录...> --output <新目录>`
不会调用 Jev，不能作为新的语义选择或闭环成功证据。

## 闭环视频默认展示统一 Representation（2026-10-16）

后续交付的 **CPU 整机闭环和 PhysX 整机闭环视频都默认展示**统一 `jev_geometry/v1`：
`mobile_platform`、`camera_manipulator` 两组 mount 系保守占用包络，
`T_world_mount`、`T_mount_camera` 两个 6D 变换（平移、RPY 和 XYZW 四元数），
以及简短的时间、来源、空间关系说明。默认每部分 3 个盒子，具体粒度跟随运行记录。

`scripts/render_agent_demos.py --runs ... --output ...` 已默认启用该区域，兼容原有
`--representation-panel` 参数。CPU 使用保存状态重建 FK；PhysX 使用保存的实测状态、
完整 `base_rotation` 和 `base_z`，保留 FK 与实测物理相机的偏差。
几何表示始终跟随物理机器人，增稳输出相机位姿独立保留。
跟踪中心到包络的距离明确标为中心距离，不当作安全净距；缺少时间戳或观测过期时显示 unknown。
独立 `ideal_floating_camera` 实验没有两个机器人模块，不虚构整机几何。

固定相机 TPR 视频继续使用 `scripts/render_tpr_demo.py <运行目录> --output <新视频>`；
CPU、PhysX 和三相机 PhysX 分支均自动追加同一几何区域，完整保留原画面。
`video/reference`、`video/execution`、`video/stabilized` 等原始 RGB 流仍供测量和评分，
最终交付通过上述渲染器生成。共同展示代码位于 `jev_agent_wbc/representation_video.py`，
不修改规划器、控制器或执行运行器。

本次六场景重录保存在
`results/joint_curves_20261016/representation_replay_20261016/final_stable_joint_curves_six_demos.mp4`，
1920×1080、16 fps、50.625 秒；使用同一 iteration14 的保存记录，原视频另行保留。
同目录 JSON、源码快照和验证记录保存来源哈希、逐帧时间线与几何一致性检查。

## 运镜验收标准更新与原 demo 重新评估（2026-10-16）

当前正式执行验收使用 `shot_acceptance/v1`，实现于 `jev_agent_wbc/shot_acceptance.py`。
它分别报告安全、主方向、整体曲线、主体构图、终帧五项结果，以及明确的失败项/证据不足项。
每项保留原有数值门槛：安全检查不变；累计反向量 orbit ≤0.003 rad、其他 ≤0.004 m；
全程总转角 <3.5 rad、最大相邻方向转角 <0.45 rad（不使用初始接续豁免）；主体入画比例 ≥98%；
终帧仍使用原有中心、面积、包围框、观察方向、轮廓轴误差和稳定保持条件。
构图是已标定主体的几何投影检查，不声称具有真实 RGB 美学评价能力。

运动完成由主方向正确、有实际运动、进度顺序正确及稳定终帧落入误差带判定，
不再要求 reference 的精确运动量或精确进度终点。相同运动类型的相邻阶段合并；
本版本对多个不同运动类型组成的复合镜头报告“未验证”，不自动判为通过。
reference 过程位置偏移、偏移累计变化、观察角度与主体尺度变化，以及旧覆盖率、
精确终点进度评分全部作为诊断。遮挡单独给出最大遮挡比例、部分遮挡和可见率低于85%的
各段起止时间、累计时长、最长连续时长；遮挡不导致验收失败，未知可见性不冒充无遮挡。
主体出画仍是构图失败，且与遮挡分别统计。结束画面检查不会因为较早的过程画面失效而连带失效。

`runner.py` 与独立相机运行器在执行结束后发布 `shot_acceptance`；主成功字段使用新验收，
旧成功标志保留为 `legacy_reference_intent_success` / `legacy_camera_intent_success`。
`executed_camera_intent` / `intent` 保留旧评分数据，并标注仅用于 reference 保真诊断。
本次只调整执行后的验收和报告，未修改在线规划、候选选择、控制器或已保存轨迹。
原规划器内部预测/排序仍使用历史 `IntentSpec`；不能把该历史字段当作新验收结论。

`results/acceptance_20261016/final/` 保存对原 `joint_curves_20261016/iteration14` 的只读复评。
安全、方向、主体构图和终帧均为 6/6；严格曲线形状为 5/6，正式自动验收 **5/6**。
上升镜头保持数值形状未通过（总转角约4.765 rad、最大相邻转角约0.888 rad）；
用户对六个视频的观感认可单独记录，不作为自动门槛豁免。历史完整 reference 评分仍为1/6。
原始日志与视频哈希保持一致，没有重新执行机器人、生成新轨迹或调用 Jev。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 \
  ros2_ws/jev_agent_wbc/scripts/reevaluate_shot_acceptance.py \
  --runs ros2_ws/jev_agent_wbc/results/joint_curves_20261016/iteration14 \
  --video ros2_ws/jev_agent_wbc/results/joint_curves_20261016/final_stable_joint_curves_six_demos.mp4 \
  --output <新的复评目录>
```

下文记录历史版本的验收口径和实验结果，不覆盖以上现行执行验收标准。

## 整体曲线与联合分配优化（2026-10-16）

本轮将整机搜索改为少量完整曲线：全局单拱曲线和保持空间形状的单调时间重参数化；
相机曲线、底盘位姿和机械臂关节通过稀疏联合求解生成，再进行独立连续 FK、整机碰撞、
自碰撞、执行器和实际终帧检查。终点在数值终帧误差带中选取，不强制回到 reference 的空间终点。
候选绑定通过检查的连续关节分配及其哈希；最新状态复核同时修正相机误差和冗余分配。
原有已发生的遮挡损失不会单独导致反复拼接新曲线；未来风险和实际跟踪失配仍会触发处理。

整机默认每次最多 6 个本地候选、最多 2 个提交 Jev 的候选。本轮实验局部数值预算 18 秒，
这是协作式软预算，单次不可中断的数值检查可能超时；它与 Jev API 推理延迟分别统计。
模型固定为 `jev-1.14.0`。原始完整意图评分保留，安全通过不等同于镜头意图通过。
曲线报告同时列出原始全程弯折指标，以及只允许起始 2 cm / 0.9 rad 接续的指标；不能混用。

最终 `iteration14` 实际轨迹的原始全程形状检查通过 5/6（上一轮 1/6）；
方向 6/6、终帧 6/6、独立安全审计 6/6。
完整原始镜头意图通过 1/6，形状改善不能当作整体任务成功。
iteration13 的重复验证暴露旧降级反馈引发多余换轨迹；最终版只在新的完整诊断及实际跟踪
均已恢复时清理这类旧反馈。最终版未在时间预算内再做第二轮重复验收。

结果、冻结源码、独立审计和最终六场景视频见 `results/joint_curves_20261016/REPORT.txt`。
视频来自实际 CPU 整机遥测，属于示意回放，不是 RGB、PhysX 或实机录像。

## 规则曲线与终点容差带实验（2026-10-15 夜间）

本轮新增通用规则曲线候选、运镜反向/弯折筛选及数值终帧容差带，不按场景名称编写避障规则。
`cinematic_geometry.py` 沿 26 个三维方向二分计算终点，保留原质心、面积、包围框、观察角及轮廓轴容差，
使用 20% 余量；这些是逐点复核的边界采样，不把它们的包围盒或凸包宣称为全部可接受区域。
`intent_paths.py` 将宽缓形变和终点偏移按几何进度参数化，保持激活前前缀和 C2 接续，
不再生成强制精确回到 reference 终点的整机恢复候选。模型可以选择误差带内的不同终点。

整机搜索先尝试规则相机曲线，再尝试经过平滑的机器人投影；平滑结果必须重新通过连续 FK、
整机/自身碰撞和执行器检查。底盘冗余运动可提前准备，反向约束针对相机运镜，不禁止底盘后退。
Jev 的选择提示和候选卡增加了方向、形状及终点容差信息。安全边界和原始镜头评分未放宽。

**目前是部分改善的实验版，不能当作六场景整体升级。** 最终轮 MoveIn 实际径向反向量从
约 9.1 cm 降为零，过程趋势与终帧通过；但 0.98 秒遮挡仍超限。Orbit 仍未找到合格新曲线，
TruckLeft 和两个 Pedestal 有退步。所有候选失败和未安装情况均保留，详见本轮报告。

结果与录像：`results/cinematic_curves_20261015/`。`REPORT.txt` 记录最终轮次、对照、限制和复现命令；
各轮 `frozen_code` 是实际运行源码，源码继续修复后的测试结果不能冒充先前录像使用的版本。
视频仍是 CPU 整机遥测示意回放，无讲解停留，不是 PhysX、RGB/SAM 或实机验证。

## TypeSafe 官方 Jev 接入与对照（2026-10-15）

`SystemOneClient` 支持官方 `https://api.typesafe.ai/v1/systemone` 的 Bearer 鉴权：
使用 `TYPESAFE_API_KEY`，或以 `TYPESAFE_API_KEY_FILE` 指定密钥文件。官方密钥只发送
到这个精确 HTTPS 地址，拒绝鉴权请求重定向；请求日志不包含鉴权头，异常和日志中的
密钥会脱敏。本机 Perplexity 兼容接口仍按原方式使用，不接收官方密钥。

官方模型文档 https://docs.typesafe.ai/models 在本轮列出新发布的 `jev-1.14.0`；
`jev-latest` 当时仍指向 1.13，因此本轮固定 1.14，并检查实际响应模型。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=ros2_ws/jev_agent_wbc \
  /usr/bin/python3 ros2_ws/jev_agent_wbc/scripts/benchmark_official_jev.py \
  --key-file /path/to/key --output /path/to/new_results
```

脚本沿用历史最终版六场景并行、pace=.03、提前量1.2秒、候选预算20秒，保存冻结源码、
逐场景 provenance、完整遥测和真实请求/响应。`replay_official_requests.py` 支持历史
完全相同的 state/questions 回放；`compare_official_jev.py` 汇总完成、安全、安装、终帧、
趋势、完整意图与 API 延迟。回放选择一致率不代表正确率；一次历史对照不能证明显著优势。

本轮结果、对照报告与六场景视频见 `results/typesafe_official_20261015/`。
视频由真实 CPU 运行日志渲染；不代表墙钟实时运行、RGB/SAM、PhysX 或实机验收。

## 统一双部分 Representation（2026-10-15）

默认整机 Agent 输入现在使用 `jev_geometry/v1`：`mobile_platform`、`camera_manipulator` 两部分，世界到安装接口、安装接口到相机的 6D 变换，以及安装接口系内的少量保守盒子。默认每部分 3 个盒子；模型文本不再包含原始关节状态、逐 link 冲突或执行器指标。

现有搜索族、配置、候选 Choice 和安装复核流程保留。候选同时绑定相机路径及两部分的位姿/占用软偏好；当前差速底盘适配器将它们转换成 WBC 次级几何目标，原相机任务与安全检查继续生效。候选预测和实际执行反馈分开记录。本阶段没有开发 Memory，也没有实现其他构型的控制器。

[接口与设计说明](REPRESENTATION_DESIGN.txt) · [JSON Schema](schemas/geometry-v1.schema.json) · [JSON 示例](schemas/examples.json) · [首次验证](REPRESENTATION_VALIDATION.txt) · [本轮原因分析与优化结果](REPRESENTATION_OPTIMIZATION.txt)

可用 `runner --representation-boxes {1,3,6}` 对比几何粒度，或 `--geometry-preferences off` 关闭附加几何偏好。六个 demo 的相同状态几何审计支持 3 盒作为紧凑性与形状保真度的折中；CPU 执行结果、候选安装和镜头成功分别报告，不能据此宣称所有动态避障任务已解决。


## 六个动态场景排查（2026-10-14）

本轮改动与逐场景实测见 [DYNAMIC_REPAIR_REPORT.txt](DYNAMIC_REPAIR_REPORT.txt)。运行结束、物理约束通过、镜头意图通过分别统计，不能互相替代。

- 整机候选搜索保留原结束时刻，增加速度脉冲、方向混合与中间幅度。先做有界几何和构图筛选，再按配置预算进行完整镜头/FK检查；CPU harness 最多检查 16 条。完整检查产出合格候选后，若累计耗时已超过 2 秒则交付，不再展开剩余分支；粗筛和单条检查仍可能超过该预算。安装仍须通过最新状态和激活期限检查。
- FK 检查尝试有限个连续关节/底盘分配，保留原精度、奇异值、速度与加速度门槛。已批准且复核通过的 FK 见证可作为 WBC 次级分配参考，不修改相机主任务与安全约束。
- LiDAR 运动估计使用过滤后的原始测量坐标；对条件良好、残差合格的圆形点云拟合球体，其他形状继续用包围盒。局部遮挡不会直接用可见点中心替代整个物体中心。球体模型是观测拟合假设，不是读取场景真值。
- WBC 同时检查相对距离速度约束；加速度屏障不相容时，可在相同执行器硬界内重解速度屏障，仍必须通过整机及自碰撞连续证书。求解器失败会区分独立 LP 确认的不可行与数值未收敛。
- CPU 调试记录额外 0.4 秒终帧确认窗口以及实际执行的意图分数；原镜头运动仍在 8 秒截止。CPU 积分与几何可见性不是 PhysX、实机或 RGB/SAM 验收。


## 整机适配器：FK 约束候选空间（2026-10-14）

`runner --planner-model floating_camera`（CPU 整机 WBC + agent 路径规划，默认配置）现已在数值工具中接入当前整机构型、速度、相机 FK 和差速底盘有效雅可比。Jev 保留原有两次 Choice；不接收新的关节矩阵、控制多面体或奇异值计算任务，也不能修改数值约束。独立 `floating_camera` 入口仍是无机器人状态的悬空相机实验。

`kinematic_feasibility.py` 实现两个层次：

- 当前状态的控制多面体包含关节位置与制动余量、速度、逐步加速度以及左右轮速度/加速度耦合。0.25 s 局部方向支撑用于调整内层候选幅度；外层候选保留，由完整轨迹检查决定是否可行，避免用短时线性化截断长期可达空间。底盘只有前进和转向两个输入，没有虚构横移自由度。
- 每条通过相机/意图初筛的候选，在截取最终候选列表前，从当前状态连续求解受约束微分 IK。默认间隔 0.04 s，检查中点与端点 FK、关节边界、轮速耦合及奇异值，包含原执行时钟内停车和 0.4 s 终帧保持。位置/姿态残差上限为 0.015 m / 0.035 rad；角速度行乘 0.35 m 特征长度后的有效雅可比最小奇异值下限为 0.015。最多 401 个时刻，预算不足明确拒绝。

初始速度优先使用实测值，缺失时明确记录 `previous_command_fallback`；CPU 运行器提供其常值控制积分模型的实际速度。只复用 WBC 的执行器不等式构造，不调用 WBC 求解、避障策略或整机 rollout。数值过程产出一条可行构型路径作为检查依据；候选获批并通过最新状态复核后，该路径也提供 WBC 的次级分配偏好，不替代 WBC 的主任务和安全求解。最新状态复核重新求解整条轨迹；安装检查构型、速度、制动能力、相机位姿、观测新鲜度和精确路径绑定。同一相机位姿的不同冗余构型不能绕过检查。

这些是采样运动学约束与局部可达性筛选，失败表示当前数值方法未找到合格连续实现；不能证明全局不可达。通过也不能替代动力学、连续时间或整机碰撞安全证明。实际避障及其失败处理仍由独立 WBC 承担。FK 检查结果存入 `kinematic_prior`、`kinematics`、候选卡 `kinematic_feasibility` 和安装检查 `kinematic_state`，保留速度来源、残差、限制利用率及首次失败时刻。

```bash
# CPU 整机 + 在线 Jev；WBC 避障保持开启。pace=.2 是慢速调试，不是实时性结果。
PYTHONPATH=ros2_ws/jev_agent_wbc OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /usr/bin/python3 -m jev_agent_wbc.runner --case all --mode jev \
  --planner-model floating_camera --pace .2 --activation-lead 1.5 --output <新结果目录>

# 可重复的 FK 正/负例，以及真实 CPU WBC 下的异步安装链路。
# 此处使用远障碍和显式触发重规划的规则选择器，不作为 Jev 或动态避障成功证据。
PYTHONPATH=ros2_ws/jev_agent_wbc OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /usr/bin/python3 ros2_ws/jev_agent_wbc/scripts/debug_fk_agent.py --output <新数值验证目录>
```

本轮调试结果与未通过项见 [FK 调试记录](results/fk_agent_20261014/VALIDATION.txt)。下节的“只用 FK 作位姿种子”描述属于此前适配器；当前整机适配器以本节为准。

## 当前默认：主体状态、运动趋势与结束画面（2026-10-14）

上一轮的相机意图规划已提交为 `29be18b`。当前在其基础上继续优化，候选生成、筛选和 Jev 选择仍只使用 floating-camera 空间，不调用 WBC rollout。独立 CPU 入口为 `python -m jev_agent_wbc.floating_camera`（安装后 `cinematic_camera`）。

原 reference 同时定义有向运动范围、过程趋势和不可被重规划覆盖的结束画面。Move-in/out 使用无遮挡投影尺度，truck 使用固定起始横轴，pedestal 使用世界竖直轴，orbit 保留相同有向观察扇区；原镜头自带的推拉、升降等伴随变化也保留。过程比较按运动进度对齐，仍不要求原时间戳上的世界位姿一致。

`subject_state.py` 接入主体 LiDAR 数值：CPU 射线返回主体与障碍的最近表面点，不输出对象 ID；估计器根据初始主体区域关联并拟合圆柱侧面。XY 与半径来自点云，Z 只有看到有效端面时才实测，否则明确依赖初始高度/位置标定。通过测量时刻相机位姿和内参计算主体质心、无遮挡投影面积、包围框、轮廓主轴及观察方向，向 Jev 发送**纯文本数值**，含来源、时间戳、有效性、分量来源和不确定性。对称圆柱的身体朝向始终为 `unknown`。测试使用模拟 LiDAR，不能当成真实传感器或 RGB 分割验证。

现有 SAM2 接口也扩展了包围框、长宽比、二维轮廓主轴和可靠尺度趋势。掩膜面积仍是可见面积；分割置信度不代表可见比例，轮廓主轴不代表三维身体姿态。`CinematicEngine` 会把参考、候选与实际主体状态按采集时刻比较，SAM2/LiDAR 偏差触发重规划反馈。没有 RGB、配准信息或可靠无遮挡条件的字段保持未知。本轮 CPU 闭环使用 LiDAR 分支，未运行 SAM2 网络或下游控制。

`camera_intent.py` 的 v2 判据将碰撞、完整有向覆盖、运动趋势和终帧分别检查。终帧要求：归一化质心误差 ≤ 0.02，面积对数误差 ≤ 0.05，包围框尺寸对数误差 ≤ 0.06，观察方向和可观测轮廓主轴误差 ≤ 3°；最后确认窗口 0.4 s，平移速度 ≤ 0.03 m/s。实际结束窗口还必须有新鲜 LiDAR 测量确认，候选预测成功不能替代执行确认。

过程约束为相同进度下最大额外位移 0.24 m、额外往返运动量 0.52 m、观察方向偏差 12°、面积对数偏差 0.28；原有反向运动、可见覆盖和阶段顺序要求继续生效。这些是当前 demo 的固定工程容差，不是通用电影规范，也不能由模型放宽。大幅绕行即使终点正确，也不能再标为完整意图成功；小幅不同世界终点仍可在画面容差内合格。

`intent_paths.py` 优先提供平滑加速/减速、交会时刻附近的小幅返回形变，以及两者的联合程序。默认空间幅度为 0.08–0.22 m，取消新候选中的大幅保持式终点；旧程序仍可精确回放并接受新判据审计。主体尺度补偿使用未遮挡几何，固定镜头内参。轨迹激活前前缀不变，边界匹配位置、速度和加速度。程序、样条几何、时间参数和父路径共同哈希绑定，迟到候选必须重新生成。

两次模型 Choice 分别选择搜索族和一条已通过工具筛选的具体候选。Jev 会看到原始终态、当前实测状态、变化趋势、候选终帧误差及额外运动量。模型文本去除了重复的几何/停留统计，完整数据仍保存于日志。已经满足全部要求的轨迹不会仅因为障碍物较近就再次退让。未知/过期主体测量、身份不符或与初始标定明显冲突会拒绝安装；当前没有一般动态主体重建模块。

安全与范围门槛沿用上轮：相机半径 0.06 m、净距至少 0.025 m，速度 1.5 m/s、加速度 4 m/s²、角速度 3 rad/s；32 个进度格仅累计可见的新进度，覆盖至少 95%，每格至少 0.025 s，最长不可用画面 0.25 s。时间可在原时长的 0.65–1.75 倍内分配，另计 0.4 s 结束确认；障碍始终按真实场景时刻推进。碰撞预测使用 50 Hz 采样与相邻线段检测，执行审计为 400 Hz；不是连续时间安全证明。

```bash
# 六个既有 demo，真实本机 Jev 文本接口；最多 16 个程序、3 个送审候选
PYTHONPATH=ros2_ws/jev_agent_wbc OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /usr/bin/python3 -m jev_agent_wbc.floating_camera --case all --mode jev \
  --pace 1 --activation-lead 2.7 --output /tmp/jev_subject_frame_new

# 规则选择器对照：--mode rule；原始参考对照：--mode strict --pace 0
# 慢速机制验证可用 --pace .3 --activation-lead 1.5；不等同墙钟实时性能
PYTHONPATH=ros2_ws/jev_agent_wbc OPENBLAS_NUM_THREADS=1 \
  /usr/bin/python3 -m pytest -q ros2_ws/jev_agent_wbc/test/test_subject_state.py \
  ros2_ws/jev_agent_wbc/test/test_camera_intent.py \
  ros2_ws/jev_agent_wbc/test/test_task_space_planning.py \
  ros2_ws/jev_agent_wbc/test/test_counterfactual_agent.py
```

运行输出包含原始意图规格、完整候选和拒绝原因、模型请求/响应、LiDAR 主体状态、执行历史、终帧确认和独立真值审计。主要结果为 `camera_intent_success`、`observed_terminal_pass`、`intent.terminal_frame`、`intent.motion_trend` 和有效覆盖。`completed` 仅表示发送完轨迹及确认窗口，不能替代镜头成功。

本轮冻结代码、失败迭代、对照数值与回放见 [主体状态闭环验证](results/subject_frame_20261014/VALIDATION.txt)。上一轮结果保留于 [v1 验证记录](results/intent_camera_20261013/VALIDATION.txt)，其“0.35 m 不同终点也通过”的判据已被 v2 终帧/趋势要求替代。

机器人适配器仍可读取当前 FK/传感器反馈作预测种子，WBC 不进入新候选工具。旧执行时钟保持不变，不发送变时长镜头；本轮没有运行 ROS、PhysX 或 WBC 闭环。这里验证的是静态标定圆柱、单球动态障碍和相机包络，不能推导整机可执行性或语义身体姿态对齐。

## 历史实现与实验记录

以下保留先前实验原文以便追溯。其中关于默认 WBC 候选预测、固定时长、reference 跟踪排序及 floating-camera 盲区的说明已被上节替代；历史结果不能作为新意图规划器的验证。

后续实验约定（2026-10-13）：WBC 避障默认开启，Agent 规划预测与实际执行均使用开启避障的 WBC。仅在用户明确要求关闭避障的对照实验时使用 `--disable-wbc-avoidance`；此前关闭避障的 demo 不作为后续实验默认配置。所有 demo 继续默认无讲解停留。

独立 ROS 2 包，用于差动底盘与六轴机械臂的运镜修复。默认 Agent 已改为 **选择修复族 → 有界整机预测 → 选择最符合意图的候选 → 最新状态复核**。WBC 持续执行并检查安全；它不等待 Jev、SAM2.1 或数学进程。原始相机轨迹、修改目标和实际执行始终分开。

论文定位、最小实验矩阵与统计方案见 [PAPER_DESIGN.md](PAPER_DESIGN.md)。候选相对排序与 Perplexity 的最新 CPU 结果见 [验证记录](results/selection_ranking_20261013/VALIDATION.txt)。此前约 1.5 小时优化记录、demo 和卡点见 [优化报告](results/optimization_20261013_0835/REPORT.md)。前一轮重构验证见 [VALIDATION.md](VALIDATION.md)；旧版历史实验仍保留在 [EVALUATION.md](EVALUATION.md)，不可当成新架构的结果。当前系统尚未达到全部动态场景的安全完成标准，也未证明 Jev 优于规则基线。

## 实验性跨阶段整机协调

对参考轨迹中存在“位置固定、姿态继续变化”的阶段，`ExecutionWBC` 自动启用 `coordination.py` 的加速度感知预测。几何检测不依赖 Pan 标签；会平移的 Pan 弧仍采用原控制分配。其余六个既有单运镜场景保持原分配。

预测最长覆盖未来 8 秒，最多 20 个速度端点，通常每 0.4 秒重新求解；初次最多 6 次、暖启动最多 3 次连续线性化。差速底盘和六个关节使用同一组变量，预测命令斜坡、速度与加速度界、轮速耦合、关节停车距离、测量/积分目标双路径及原预算末端停车。实际速度可作为初始状态；缺失时明确记录使用上一条命令。未来执行响应仍采用运动学近似，不能视为已标定的 PhysX 或实机动力学。

快速 QP 也独立保留原预算停车界，即使异步预测缺失也逐步收敛至零命令。当前相机任务保留在快速 QP 中，未来分配替换会与其冲突的瞬时朝向/舒适姿态目标；失去预测时使用当前约束内的低底盘速度偏好。没有后置限幅，也没有新增任务输入、重定时或评估容差。现有避障、关节制动界及扫掠证书继续独立检查实际命令。未来预测不包含障碍物约束，日志明确 `coordination_certificate=False`；它不是未来碰撞安全或镜头可行性的证明。

实时 ROS 使用独立进程：一次只允许一个预测任务，过期超过 0.6 秒、目标版本不一致或偏离预测状态的结果拒绝安装。CPU/PhysX 默认同步执行用于数值复现；同步预测可能耗时数百毫秒至数秒，不能宣称达到 50 Hz。PhysX 可添加 `--async-coordination` 验证非阻塞路径。原 `--allocation-preview` 是旧分配器消融路径，不叠加新的协调器。

新增日志包括预测时间范围、实际/命令速度初值来源、预测残差与关节余量、相机速度残差、实际速度计算的关节制动余量、异步预测接受/拒绝次数。制动余量是给定操作减速度下的诊断量，不是鲁棒可行域保证。

回归：`PYTHONPATH=ros2_ws/jev_agent_wbc OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -m pytest -q ros2_ws/jev_agent_wbc/test`。新增测试覆盖跨阶段时长不变、下一阶段改变当前分配、低转向加速度、命令斜坡导数、预测不能越过硬约束和制动、异步过期/目标版本/状态偏差拒绝及既有单运镜路径不启用。

## 外部输入与运行时证据

`task.json` 仍只接受四个字段：

```json
{
  "reference_camera_trajectory": {
    "frame": "world_z_up_camera_link",
    "samples": [{"t": 0.0, "position": [0.16, 0.0, 0.91], "quaternion_xyzw": [0, 0, 0, 1]}]
  },
  "shot_phase": [{"name": "develop", "motion": "movein", "start_s": 0, "end_s": 8}],
  "subject": {"id": "blue_subject", "description": "blue cylinder with a yellow top"},
  "intent": "Approach the subject for increasing emphasis."
}
```

这是字段示意；合法轨迹需要至少三个时间递增的采样，阶段连续覆盖原始预算。世界系 Z 向上，相机 link 的 X 向前、Y 向左，四元数 XYZW。不能把机器人配置、障碍未来真值或视觉容差加入任务接口。

运行时使用关节、里程计、LiDAR 地图、主体三维跟踪及可选 RGB 分割。`runtime_scene.json` 仅用于测试环境生成与独立评分；规划器只接收传感器观测及其局部运动估计。主体初始化、相机内参及 TF 是运行时标定，不是新增的拍摄输入。

## 默认决策流程

1. 数学进程在最多 5 秒窗口内诊断原参考的碰撞、关节、速度和执行困难；最多 9 个廉价 Jacobian 探针提供局部方向证据，不承担轨迹可执行性证明。新冲突、观测问题或恢复事件触发慢速决策。
2. Jev 用一次 Choice 选择半径/高度、横移/高度、联合修改或恢复搜索族。这里没有运动授权。
3. 最多 8 个提案做粗筛，最多 3 个分支做完整 WBC 预测。相同激活前缀只计算一次，粗筛在激活之后的最多 2.4 秒内进行。候选相对不可变原参考构造 C2 变化；初版修复族主要搜索平移并补偿原始主体构图，不是任意六维或拓扑规划器。
4. Jev 用一次无拒绝选项的 Choice，从已经经过程序筛选的候选中选择最符合当前意图的一条；候选为空时不调用模型。输入包含预测**实际执行**的构图、距离尺度代理、观察角、阶段进度、最差时刻和后续余量；不发送点云或图像，不做六轮数值权限搜索。
5. 批准绑定候选哈希、原参考和父方案。数学进程从新快照重预测；控制侧只做有界的状态、占用、尺寸/速度变化与证据门禁。迟到方案不能偷偷改变生效时刻。
6. 执行反馈可触发新的取舍。底层仍独立通过约束与动态扫掠证书，无法认证时报告失败。镜头时间不会暂停或延长。

当前 API 默认为 `http://127.0.0.1:8000/v1/systemone`、`model="perplexity"`（可通过 `--model` 显式切换），采用本机服务的 Choice 协议。候选使用短别名，精确轨迹及别名到哈希的绑定保留在日志中。模型概率只记录为决策信息，不作碰撞或传感器可信度证明。

`Envelope` 仍是数值路径构造器的边界，但搜索生成的 Envelope 明确标为未批准。主路径只有 `Approval` 才可安装。旧 `SemanticDecider` 与 `optimize_envelope` 保留供历史回归使用，默认 Engine 不调用它们。

## 计算与验证边界

| 项目 | 默认配置及含义 |
|---|---|
| WBC | 原有 50 Hz 控制配置，不等待 Agent |
| Agent 事件 | 最小间隔 0.5 s；实际完成频率取决于网络与数学预测，不能称为 2 Hz 吞吐 |
| 激活提前量 | 4 s，参数 `--activation-lead`；短预算不够时拒绝，不重定时 |
| 数学预算 | 8 个粗筛、3 个完整分支，0.1 s 远期预测，共享前缀；安装复核前 1.2 s 使用 0.02 s 步长 |
| 预测窗口 | 最长 8 s；长镜头的修改在窗口末平滑恢复，窗口之外不宣称已验证 |
| SAM | 独立进程，默认最多 4 Hz，一次仅一个未完成任务 |
| 决策超时 | 每次 API 请求默认 3 s；超时期间底层继续运行 |

完整预测仍可能耗时数秒，这是当前局限。`--pace 1` 只是仿真/墙钟速率的上限，计算过慢时仿真仍可能落后；不能据此宣称在线实时。`--pace .2` 只用于明确标注的慢速机制调试。必须同时报告墙钟时间、API 延迟、端到端延迟、控制 p95/p99 和未安装原因。

五个验证结果分开记录：`physical_valid`、`target_authorized`、`predicted_visual_valid`、`state_fresh`、`observed_visual_valid`。它们可为 pass/fail/unknown。遮挡未知、缺内参、失去分割都不能记成通过。已批准的已知证据后来丢失会要求重新审批。构图中心一致也不能单独证明全部视觉目标满足。

最新预测与批准结果之间的数值一致性余量集中在 `SearchBudget`，不是 Jev 生成的语义容差。不得用它们扩大权限或修改固定评估规则。正式统计前需在开发集校准并冻结。

## SAM2.1 与 LiDAR 接入

`perception.py` 接收带 subject ID 和采集时刻的简洁证据，区分当前有效、历史可信和当前未知。最多 64 个几何记录用于匹配迟到 RGB 的采集时刻，最新几何仍单独用于规划；新几何失效时不能靠历史记录恢复“当前有效”。可见 mask 面积与主体投影尺度分开；遮挡、截断、时间不同步或几何投影不一致时，不把面积损失当成径向补偿依据。未来可见性和完整投影尺度目前仍为 unknown。

ROS 输入：`/joint_states`、`/base_controller/odom`、`/lidar/points`、`/lidar/front/points`、`/cinematic/subject_pose`。可选 `/camera/camera_info`、`/cinematic/subject_visual`。主体 PoseStamped 必须在 world/odom 中且由独立跟踪器完成身份关联；默认专用 topic 对应任务中的一个主体。相机 TF 按 RGB **采集时刻**查询，默认 `camera_frame=camera_link`，坐标约定须匹配该 link。LiDAR 已在各自采集时刻转到世界系。

SAM 观察器使用官方 SAM2.1 图像预测器：人工初始化框，后续用前一有效 mask 框提示当前 RGB 帧，固定内存、不读取未来帧。质量下降返回 unknown；它没有实现一般主体重识别或完整视频记忆传播。置信字段是 SAM 的预测 IoU，不能当作身份概率。初帧冷启动结果可能过期，主节点会按采集时间拒绝使用。

现有 SAM 环境是 Python 3.11，ROS Humble 是 Python 3.10。独立观察器通过有界进程管道调用现有 SAM 环境，不修改两者依赖：

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH="$PWD/ros2_ws/jev_agent_wbc:$PYTHONPATH"
/usr/bin/python3 -m jev_agent_wbc.sam_observer --ros-args \
  -p sam_python:=/home/w84384651/miniconda3/envs/sam2/bin/python \
  -p checkpoint:="$PWD/sam21/checkpoints/sam2.1_hiera_base_plus.pt" \
  -p subject_id:=blue_subject \
  -p image_topic:=/camera/image_raw \
  -p seed_box:='[100.0, 70.0, 220.0, 210.0]' \
  -p observation_rate_hz:=4.0
```

示例框必须换成实际初始主体框，且图像与控制使用同一时钟；仿真另加 `-p use_sim_time:=true`。先启动观察器预热，再启动拍摄。人工框是运行时主体初始化，不是语义策略或评价阈值。

SAM 推理限频和结果投递分开：最多 4 Hz 启动推理，以最多 20 Hz 的轻量定时器取回已完成结果，不增加推理队列。日志分别记录推理加 IPC、投递等待及总观察耗时。出画契约按持续时间比较，不按预测采样点数量比较；零修改选择不计为安装。

Isaac 运行器已将当前 LiDAR 主体观测送入 Agent，并使用渲染相机内参。当前无 ROS 的 Isaac 路径仍采用独立离线 SAM 评分，没有把离线视频或未来 mask 喂给在线决策。要评估在线分割收益，应运行 ROS 感知闭环或另加经过验证的实时帧适配器。

## 运行与复现

仓库根目录：

```bash
# 必需回归
PYTHONPATH=ros2_ws/jev_agent_wbc OPENBLAS_NUM_THREADS=1 \
  /usr/bin/python3 -m pytest -q ros2_ws/jev_agent_wbc/test

# CPU 整机闭环；rule 是阶段/历史/后续余量数值基线
PYTHONPATH=ros2_ws/jev_agent_wbc OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /usr/bin/python3 -m jev_agent_wbc.runner --case orbit --mode rule --pace 1 \
  --agent-period .5 --activation-lead 4 --candidate-budget 3 --output /tmp/jev_rule_new

# mode: strict / rule / jev / both；both 为 strict + jev
# PhysX 物理审计与 RGB，输出必须使用新目录
source /opt/ros/humble/setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
  /usr/bin/python3 ros2_ws/jev_agent_wbc/scripts/jvs_isaac_bootstrap.py \
  --isaac-python /home/w84384651/miniconda3/envs/env_isaaclab/bin/python \
  --runner ros2_ws/jev_agent_wbc/scripts/run_isaac.py -- \
  --case orbit --mode both --physics-hz 1000 --pace 1 --record --output /tmp/jev_physx_new

# Isaac 退出后独立离线评分，包括中止运行的已执行前缀
/home/w84384651/miniconda3/envs/sam2/bin/python \
  ros2_ws/jev_agent_wbc/scripts/evaluate_sam2.py --runs /tmp/jev_physx_new \
  --sam-root "$PWD/sam21"

# 相同候选集的选择回放，不能代替闭环性能实验
/usr/bin/python3 ros2_ws/jev_agent_wbc/scripts/replay_decisions.py \
  /tmp/jev_physx_new/orbit_jev/events.jsonl --selector rule --output /tmp/jev_replay_new
```

六组既有场景仍为 orbit、movein、truck_left、truck_right、pedestal_up、pedestal_down，每组 8 秒。内部 establish/develop/resolve 是同一运镜的阶段；它们不等价于已验证 Orbit → Pan 多运镜组合。新论文的组合与意图对照场景见设计文档，尚未完成正式数据集。

已有 ROS 机器人/模拟器可构建并运行：

```bash
source /opt/ros/humble/setup.bash
colcon --log-base ros2_ws/jev_agent_wbc/.colcon_log build \
  --base-paths ros2_ws/jev_agent_wbc --packages-select jev_agent_wbc \
  --build-base ros2_ws/jev_agent_wbc/.colcon_build \
  --install-base ros2_ws/jev_agent_wbc/.colcon_install --symlink-install
source ros2_ws/jev_agent_wbc/.colcon_install/setup.bash
ros2 launch jev_agent_wbc cinematic_ros.launch.py task_file:=<绝对路径>
```

只有 `/isaac_joint_commands` 一个执行发布器。`scripts/smoke_ros.py --output <新目录>` 使用专用 DDS 域的合成传感器，仅验证传输与控制接口。

## 文件与独立指标

- `engine.py`：事件状态机、异步数学/语义任务、观测反馈与安装。
- `candidates.py`：预算、精确候选、共享前缀、结果契约及最新状态门禁。
- `perception.py`、`sam_observer.py`：小型证据接口、时序/投影一致性及可选独立 SAM 进程。
- `jev.py`：类型化 Choice、精简后果卡、规则基线；不生成自由 JSON 动作。
- `planning.py`、`reference.py`：整机 rollout、不可变参考、C2 修改。
- `safety.py` 与 `dynamic_*`：原有独立安全约束与扫掠检查。

每轮日志保留 `strategy_requested`、`candidate_set`、`selection_requested`、`candidate_approved`、`plan_installed`、拒绝原因与观测反馈。`jev_api.jsonl` 记录实际请求、原始响应、HTTP 详细错误与延迟。初始化事件保存预算，运行器保存任务、环境和源码指纹。成功、拒绝、未结束决策分开计时。

四个固定指标仍是：离线 mask 质心构图误差、相对 mask 面积尺度误差、有效主体三维坐标系中的观察角误差，以及原始物理时刻采样的平移 jerk/角加速度。缺失不记零，不做动态时间规整，不以几何指标代替人类意图判断。新增延迟分位数不改变四项评分协议。

静态主体测试场景的无遮挡参考视频在执行结束后按原始相机位姿单独渲染，避免执行/参考间的可见性切换污染。该协议不支持未经建模的移动主体回放。`reference_rendering.json` 记录帧数与协议；评分器标记 mask 面积突变和缺失供人工复核，不自动排除帧或修改误差。旧版 MoveIn 参考曾受前景污染，不可使用其视觉均值比较方法。

物理侧检查 `completed`、`safety_pass`、`within_budget_success`、接触、净距和子步约束；中止前未碰撞不代表整个任务成功。当前存在 PhysX 无认证控制、加速度/关节余量及预测一致性问题。LiDAR 为稀疏理想射线或 ROS 外部输入；未证明未知环境、通用移动主体、自碰撞全局覆盖或实体硬件安全。

模型身份更正与四模型实测见 [模型对照](results/optimization_20261013_0835/MODEL_COMPARISON.md)：旧 `perplexity` 别名实际返回 Perplexity Decider，历史 `mode=jev` 不能当作 TypeSafe Jev 结果。


## 候选排序与 CPU demo（2026-10-13）

第二次调用使用独立的 `SELECTION_POLICY`，问题为“这些候选已经经过程序筛选，请选择最符合当前意图的一条。”选项只有 `c0`、`c1` 等实际候选，没有 `refuse`。第一次搜索策略仍使用原来的 `CANDIDATE_POLICY`，每次进入策略阶段都会恢复该提示。选择结果仍绑定原候选哈希，物理约束、最新状态和预测一致性复核保持启用。空候选集不发送请求；模型返回集合之外的值视为协议错误。

客户端、Engine、CPU/Isaac 运行器及决策回放的默认模型统一为 `perplexity`，可用 `--model` 显式覆盖。本机服务返回的模型名称为 `perplexity-decider-v1.1`，以每次 API 日志为准。

本轮验证与录像位于 `results/selection_ranking_20261013/`：`live_cpu` 使用默认速度上限和 4 秒激活提前量；`slow_cpu` 使用 0.2 倍速度上限和 1.5 秒提前量作机制诊断，两组不混为实时性能结果。CPU 延续既有整机数值积分与虚拟悬空相机示意回放，未替换为自由飞行相机控制器。完整结论见该目录的 `VALIDATION.txt`。

新运行可以直接沿用现有 demo 布局录制，输出路径必须尚不存在：

```bash
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 ros2_ws/jev_agent_wbc/scripts/render_agent_demos.py \
  --runs <运行目录1> <运行目录2> --output <新目录>/demo.mp4
```

所有后续 demo 默认按仿真时间连续回放，不插入中途讲解停留、事件定格或章节结果停留；候选和结果讲解仅保存为独立预览图。

每段保留三视图、原始/实际/候选轨迹、模型输入输出、复核与安装状态；同名 JSON 保存来源哈希、事件时间线和视频校验和。录制结果经过完整解码检查，关键帧保存在同目录的 `_previews` 文件夹。旧 demo 文件未覆盖。

## 独立悬空相机与 WBC 避障消融（2026-10-13）

`floating_camera` 是独立执行模型：只有相机位置和朝向，没有底盘、关节状态、机器人 FK、WBC、执行器限幅或局部避障干预。每个控制时刻直接使用当前 Agent 目标的位姿，达到 1:1 task-space 跟踪。默认相机碰撞体为半径 0.06 m 的球体；预测用 50 Hz 路径加相邻采样之间的相对线段最近距离，独立真值评分以 400 Hz 采样。它不是连续非线性路径或实物安全认证。

该诊断沿用两次 Jev Choice、相同任务空间修复族与精确候选绑定；把整机 rollout 换为理想相机路径检查。观测器仅适用于当前单球障碍夹具：从当前相机射线命中点拟合球体，从过去 0.8 秒的中心估计速度，不向规划器提供未来关键帧。视觉使用原任务的主体标定，未接入 RGB/SAM。候选生成异步运行；镜头预算不延长。

```bash
PYTHONPATH=ros2_ws/jev_agent_wbc OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /usr/bin/python3 -m jev_agent_wbc.floating_camera --case all --mode jev \
  --pace .2 --activation-lead 1.5 --output <新的独立相机结果目录>
```

CPU WBC 运行器新增 `--disable-wbc-avoidance`：使用同一套 shot-aware 相机任务、姿态偏好、关节余量与执行器约束，跳过静态/动态碰撞约束、障碍相关的跟踪权重退让、撤退、避障制动和逃逸回退；扫掠计算只作为观察值，不改变命令。执行器不可行仍报错。候选预测同步使用该控制模式，但独立碰撞筛选仍会淘汰预测碰撞的候选。关闭避障不意味着候选为空时会自动得到一条安全路径。

```bash
PYTHONPATH=ros2_ws/jev_agent_wbc OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /usr/bin/python3 -m jev_agent_wbc.runner --case all --mode jev \
  --pace .2 --activation-lead 1.5 --disable-wbc-avoidance --output <新的WBC结果目录>
```

`--target-replay-root <独立相机结果目录>` 与 `--mode strict` 联用，可将源 Agent 已安装的目标按相同哈希、安装时刻和激活时刻交给 WBC，分别打开/关闭避障。此处 strict 只关闭新的语义调用；复用的是修改后的目标，不是未修改的原始轨迹。日志记为 `target_replayed`，不冒充新安装或新模型决策；为了隔离执行器差异，固定目标回放不重新进行 WBC 安装审批。所有碰撞均独立记录。

本轮数据位于 `results/independent_camera_20261013/`，包括独立相机、原轨迹对照、相同目标的 WBC 开/关对照、关闭避障的在线 Agent 运行，以及 `demo/` 视频。默认 ROS/PhysX 配置仍启用原有避障；本轮关闭操作通过上述 CPU 开关显式选择。独立相机避障通过不等于整机扫掠安全或原始镜头完全达标，详见该目录 `VALIDATION.txt`。

### 规划避障开启、执行避障关闭的 CPU 对照

`--disable-wbc-avoidance --planning-wbc-avoidance` 将两个控制器分开：Agent 的困难诊断、候选完整轨迹筛选和安装前重新预测使用开启避障的 WBC；实际执行使用关闭避障的 TrackingOnlyWBC。预测始终从实际观测状态和上一条实际命令出发，独立碰撞检测仍保留。未指定 `--planning-wbc-avoidance` 时，规划沿用执行端的避障设置，兼容原有对照。

```bash
PYTHONPATH=ros2_ws/jev_agent_wbc OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 -m jev_agent_wbc.runner \
  --case all --mode jev --pace .2 --activation-lead 1.5 \
  --disable-wbc-avoidance --planning-wbc-avoidance \
  --output <新运行目录>
```

该配置有意采用不同的预测和执行控制器；规划通过不能证明关闭避障后的实际执行安全，需查看闭环真值净距。`summary.json` 和初始化事件分别记录 `planning_obstacle_avoidance_enabled` 与 `obstacle_avoidance_enabled`。新 demo 名称为 `agent_only_wbc_avoidance_off.mp4`，保持六场景布局及无讲解停留的连续回放。

### 在线规划模型对比（执行 WBC 默认避障开启）

运行器的 `--planner-model wbc` 是原有默认完整框架；`--planner-model floating_camera` 仅替换 Agent 的诊断、候选筛选与安装前预测。两者均接收实际机器人状态、同一套在线 LiDAR 跟踪观测和实际执行历史，实际运动由开启避障的 WBC 积分产生，绝不把候选位姿直接写入机器人。Perplexity 两阶段策略/选择、精确轨迹绑定、原始时钟和新鲜度检查保持一致。

相机规划分支采用半径 0.06 m 球体，不预测底盘/机械臂：每次预测以实际 FK 相机位姿播种，并用五次平滑函数在激活提前量内收敛至候选目标；这是假设的理想跟踪过渡，不能当成实际 WBC 运动。使用同一观测 surfel 位置、速度及动态不确定性，按 50 Hz 相机采样与相对线段最近距离筛选；安装时重新核对实际相机位姿和地图变化。执行偏差再次进入下一轮 Agent。没有未来场景真值、RGB/SAM 或完整整机安全证书。

```bash
PYTHONPATH=ros2_ws/jev_agent_wbc OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 -m jev_agent_wbc.runner \
  --case all --mode jev --pace .2 --activation-lead 1.5 \
  --planner-model floating_camera --output <相机规划在线闭环目录>
# 将 floating_camera 改为 wbc，在另一个新目录运行完整 WBC 预测对照。
```

本轮对比数据在 `results/online_planner_comparison_20261013/`。两组各六场景重新在线调用模型、顺序运行，保持相同场景/时长/候选预算/提前量/速度配置。视频沿用三视图布局且无讲解停留。CPU 在线闭环指执行中异步规划并消费实际反馈，`pace=.2` 不是实时性能承诺；单轮六场景对比不构成统计显著性结论。

## TPR-WBC：轨迹保持的被动避障（2026-10-13）

当前默认避障已替换为 **严格任务分层 QP＋短时构型预测**。Agent 决定相机轨迹修改；WBC 在收到的六维相机任务下协调底盘与机械臂，取消提前撤退、逃逸朝向和障碍相关相机权重退让。旧算法文件仅保留作历史/显式对照，不参与默认执行；旧 `allocation_preview` 开关不再切回旧控制器。上文历史对照中的“原有避障”描述的是当时版本。

动态相对运动约束、关节停止余量、自碰撞、轮速和实测加速度共同进入安全集合。任务保持不可行时明确上报 `degraded`，不会算作正常跟踪成功；安全集合不可行时报告失败。ROS 使用异步构型预测，并要求新鲜且时间戳接近的底盘/关节实测速度。

固定相机、飞行障碍 demo 的算法说明、复现命令和模型边界见 [TPR_WBC.md](TPR_WBC.md)。结果目录为 `results/tpr_wbc_20261013/`，其中 `final_kinematic_72s/demo.mp4` 是三圈连续运动学闭环。物理版使用 `physx_render_corrected_48s/demo.mp4`；旧 `physx_48s`、`physx_refined_48s` 的障碍物显示未同步 PhysX 位姿，视频中的球错误地静止，不能用作动态障碍视觉演示。新版逐物理步读回碰撞球位置，并逐帧核对画面中的球心。完整指标及未通过项见该目录 `REPORT.txt`；运动学通过不能替代物理约束审计。

## 状态条件化搜索、意图取舍与构型准备（2026-10-14）

默认整机 Agent 路径现在包含三个结构化调用，仍属于两个决策阶段：第一阶段先选搜索族，再依据当前机器人局部能力探测、上一轮失败区段和剩余预算选择搜索配置；第二阶段选择具体候选。搜索配置选择幅度种子与连续冗余分配顺序，不允许修改碰撞余量、真实关节/执行器限制或 FK 验证阈值。局部能力探测是带耦合执行器限制的方向支持 LP；其平移支持未固定相机朝向，不能当成完整可达性证书。

新增 `executable` 数值行为：从真实状态、速度和当前相机轨迹出发，求解带整机/自身碰撞、加速度、轮速和关节限制的连续机器人运动；将这一运动的 FK 结果作为**显式的新相机候选**，再独立严格复核该相机曲线。投影过程可以偏离原始镜头目标，但不能直接作为已通过的候选。生成后保存的是有哈希、激活时刻、原参考绑定的 `fk_camera_program/v1`，包含用于连续求解的机器人初始轨迹。安装后，WBC 的相机任务仍保持严格优先级；不能继续隐式投影已安装目标。

候选的整机碰撞检查使用当前观测及其运动外推，并保留连续控制区间的扫掠下界、自碰撞包络、FK 残差、关节及执行器检查。检查失败区分物理碰撞、物理限值和数值搜索未验证通过。搜索预算耗尽会记录未检查提案数，不声称该搜索族无解。预算退出是完整分支之间的协作式检查，单个分支可能超过所设墙钟预算；激活过期检查仍独立拒绝迟到候选。

镜头意图不再是默认相机候选的硬拒绝条件。可见覆盖缺失、过程偏离、终帧不一致保留原始测量、区间、后果及恢复未知标记，交给最终选择阶段。候选评分使用连续 FK 实现的预测相机运动。执行结果仍对照不可变原始 reference 审计，安装了降级轨迹不等于原镜头成功。已安装候选若仍缺失原镜头意图，会继续请求恢复；恢复候选显式回到不可变原始终帧，过去丢失的可见覆盖仍保留。

构型准备是已安装相机轨迹前段的冗余分配，不插入等待、关节跳变或额外相机动作。准备计划绑定轨迹和最新验证状态，按后续整段预测的关节制动余量与速度余量寻找最早合格状态：关节制动/位置余量至少 0.08 rad、速度和轮速不超过上限的 98%；准备过程相对激活状态的关节变化不超过 0.45 rad，底盘平移不超过 0.20 m，转向不超过 0.40 rad。当前已合格可取零时间；找不到则明确报告 `not_found`。报告中的最短时间仅指当前预算内选中连续分配轨迹上的最早已验证采样时刻（通常 0.04 s 分辨率），不声称全局时间最优。

WBC 从批准安装后开始在相机任务和硬约束内跟随复核过的冗余分配，相机轨迹仍只在其精确激活时刻改变；保留滚动预测与持续余量维护。实测进入准备目标且余量也满足要求，才记录完成；跟踪不可行、余量不足和未确认进入分别记录，不能把预测准备完成当成实测成功。生成阶段额外保留关节、速度、加速度和轮速余量，供实测状态纠偏使用；这些限制只收紧生成问题，不放松执行硬条件。

复现本轮慢速 CPU 闭环（新目录）：

```bash
PYTHONPATH=ros2_ws/jev_agent_wbc OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /usr/bin/python3 -m jev_agent_wbc.runner --case all --mode jev \
 --planner-model floating_camera --pace .05 --activation-lead .8 --output <新目录>

PYTHONPATH=ros2_ws/jev_agent_wbc OPENBLAS_NUM_THREADS=1 \
 /usr/bin/python3 ros2_ws/jev_agent_wbc/scripts/audit_state_search.py \
 --runs <运行目录> --output <审计目录>
```

`floating_camera` 在此是历史保留的规划器参数名：实际执行仍是整机 WBC，候选包含整机 FK 与扫掠检查，不是自由悬空相机执行。`pace=.05` 是机制调试速度上限，不能宣称实时；视频按仿真时间连续回放。默认本机服务仍请求 `perplexity`，真实返回模型名称以 `jev_api.jsonl` 为准，不把它冒称为另一个模型。CPU 检查不代替 PhysX 或硬件验证。

本轮结果、逐场景失败、准备对照、测试与 demo 见 `results/state_conditioned_20261014/REPORT.txt`；主演示运行目录为 `budgeted_runs/`（8 秒协作式候选预算、1.2 秒激活提前量），`release_runs/` 保留上面命令对应的 2 秒预算对照。两组计算预算不同，不构成等预算算法消融。主演示可通过结果目录中的 `run_budgeted.py --output <新目录> --candidate-seconds 8 --lead 1.2 --pace .05` 复现；Python 环境同上。旧日志与旧视频全部保留。
