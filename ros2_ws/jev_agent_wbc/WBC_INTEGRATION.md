# 五款机器人 WBC 检查与避障 Agent 接口

检查日期：2026-10-20。

本轮按照给定选型检查实际代码、依赖、控制权和坐标系，并实现可验证的适配。
原有平地运动通过记录仍保留；它们不能代替新控制器、相机连续跟踪或避障验收。
每款机器人的当前结果写在其 `WBC_READINESS.json`、`WBC.md` 中。

## 本轮收尾结果

用户最终将 G1、Go2 的本轮目标调整为“基础稳定控制与接口适配完成”，降低精密停车要求。本轮按此范围完成收尾并停止优化；原精密验收结果未改写。Husky、Ridgeback、TIAGo 保留已通过的原控制验收。
所有新试验均使用新目录，第一阶段源码及记录保留于
`validation/wbc_stage1_sources_20261020/`。

| Robot | 新控制器实际物理证据 | 范围与边界 |
| --- | --- | --- |
| Husky＋UR5 | 修正重复地面、保留原摩擦材质，真实 OCS2 显式终点保持。`ocs2_acceptance_03`：163.16 秒完整六方向、独立六轴臂、全部 10 DOF 逐段停车、实时与原始渲染均通过，零安全违规；相机 RMS 15.04 mm、峰值 32.15 mm、40/2100 帧超过逐点 30 mm，最终误差 0.592 mm（原验收使用 RMS＋终点）。[物理](validation/wbc_husky_full03_physical_independent_20261020.json)、[完整套件](validation/wbc_husky_full03_suite_independent_20261020.json)、[关节限值](validation/wbc_husky_full03_joint_limits_independent_20261020.json)、[渲染](validation/wbc_husky_full03_render_independent_20261020.json)独立复核通过 | 仅该版本记录场景与任务通过；感知覆盖和环境避障仍未认证 |
| Ridgeback＋UR5 | 单地面场景下的 `ocs2_acceptance_02`：真实全向 OCS2，122 秒完整八方向、独立六轴臂、全部 9 DOF 逐段停车、实时与原始渲染通过，零安全违规；相机 RMS 3.35 mm、峰值 15.05 mm，2100 帧无逐点超容差，最终 3.34 mm。[物理](validation/wbc_ridgeback_full02_physical_independent_20261020.json)、[完整套件](validation/wbc_ridgeback_full02_suite_independent_20261020.json)、[关节限值](validation/wbc_ridgeback_full02_joint_limits_independent_20261020.json)、[渲染](validation/wbc_ridgeback_full02_render_independent_20261020.json)独立复核通过 | 仍为原平面关节底盘，轮地动力学和感知避障未认证 |
| TIAGo | 真实 World/CartesianPose/Statechart/Executor/QPController，加 Giskard/FCL 安全投影、实测轮驻车和独臂归位。修正重复地面后，`giskard_executor_full04`：129.43 秒完整六方向、torso＋七轴臂独立运动及归位、全部 14 DOF 逐段停车、实时和原始渲染均通过。相机 RMS 10.23 mm、峰值 16.99 mm，1600 帧均在原 30 mm/0.08 rad 内；[物理](validation/wbc_tiago_full04_physical_independent_20261020.json)、[完整套件](validation/wbc_tiago_full04_suite_independent_20261020.json)、[关节限值](validation/wbc_tiago_full04_joint_limits_independent_20261020.json)、[渲染](validation/wbc_tiago_full04_render_independent_20261020.json)独立复核通过 | 仅该版本记录场景与任务通过；感知覆盖和环境避障仍未认证 |
| G1 | AGILE height80/12＋真正 Pink15；实际质量/COM 重力补偿、完整任务绑定与 60 ms 指令租期接入物理闭环。`basic_full_validation_01` 完整 92 秒，九项方向/拍摄、原相机 RMS/终点要求、4500 个新鲜 FCL 检查及基础停车全部通过。[基础层独立复核](validation/wbc_g1_basic_full01_independent_20261020.json)内含原物理、关节限值与反馈检查；[实际末帧渲染](validation/wbc_g1_basic_full01_render_independent_20261020.json)通过 | 原精密速度 8/9、停车 4/9；左转指令 0.12 rad/s、实测 0.1520 rad/s，最大末半秒关节跨度 20.24 mrad。4 次瞬时求解失败原样记录；未触发物理保护。精密资格、全部下身驱动力矩、环境/感知避障未认证 |
| Go2＋Piper | LeggedManip WBC210/18；修正移动相机采样、mixed-frame/高度反馈与新任务输入连续性，停车采用新鲜 XY 速度反馈。`adaptive_suite_planar_brake_01` 完整 92 秒，九项速度/相机、4500 个新鲜 FCL 检查、源限值及基础停车全部通过，零不可行反馈。[基础层](validation/wbc_go2_planar_basic01_independent_20261020.json)、[实际 XY 输入](validation/wbc_go2_planar_input01_independent_20261020.json)和[实际末帧渲染](validation/wbc_go2_planar_basic01_render_independent_20261020.json)独立复核通过 | 原精密停车 7/9；左转和移动拍摄仍有残余摆动，最大末半秒关节跨度 25.07 mrad、相机位置跨度 11.17 mm。不承诺任意六维轨迹、全姿态视野或环境避障；旧版本六维成功记录不转移到本配置 |

已停止继续调参。失败试验、原始 QP、旧源码及原精密门禁均保留，未将未通过的新增跟随约束或未物理运行的 Ki4 实验设为默认。G1 缺失运动回包时保持最后实际写入目标、速度置零，并按每个物理子步的新状态计算独立静态重力；[实际逐写入真 Pin 复核](Robot_TIAGo/wbc/results/g1_gravity_hold_physical_contract01/summary.json)通过。恢复墙钟证明实验未接入验收，基础报告也不声称已认证恢复墙钟上界。

两款足式机器人使用各自的 `run_basic_wbc.sh`。薄入口保留原运行器的严格退出规则，原精密失败仍可返回 2；基础结果由 `scripts/audit_wbc_basic.py` 单独判定。它复用独立原始任务、关节限值与反馈检查，保留原相机 RMS/终点容差和原物理保护，仅将本轮基础要求明确为：

- 九项原任务的请求方向均有至少 0.01 m/s 或 0.01 rad/s 的实测响应；实际速度误差仍逐项报告，不授予精确速度伺服资格。
- 末 0.5 秒底盘位置/旋转跨度不超过 10 mm/0.03 rad，相机不超过 20 mm/0.05 rad，全部旋转关节不超过 0.05 rad，最终全轴线速/角速不超过 0.03 m/s、0.05 rad/s。
- 原配置/速度限值、新鲜实测自碰撞、未中止与完整实时运行检查继续执行。公开驱动或环境状态若未知，仍保持 unknown；相机末帧方向渲染不能代替全姿态感知覆盖。

公共物理层位于 `jev_agent_wbc/wbc_physical.py`，包含实际碰撞形状反馈、不可变绝对时钟任务、约束时效及全轴停车检查。
`scripts/audit_wbc_physical.py` 独立重算原始任务插值、光学坐标转换、接纳观测绑定、全部关节停车、时序与已知约束失败。
`scripts/audit_wbc_suite.py` 检查全方向覆盖及每段完整停车；`scripts/audit_wbc_limits.py` 从原始全部关节位置/速度及足式物理子步峰值重新检查限值，不信任控制器自报通过。
`scripts/audit_wbc_feedback.py` 另查公共误差标志、布尔/null、两部分剩余能力，以及 G1 实际执行回包的完整任务哈希、序号、整个 20 ms 执行区间及四个 5 ms 物理子步的原 60 ms 双时钟租期；它不替代物理验收。保护触发而未完成停车时，独立复核仍保留未完成状态及原生安全失败。
原精密验收的末 0.5 秒停车门槛保持：底盘/相机位置跨度小于 5 mm、旋转跨度小于 0.01 rad、旋转关节跨度小于 0.01 rad、移动关节跨度小于 2 mm，最终全轴线速小于 0.02 m/s、角速小于 0.03 rad/s。
未知环境约束、未验证传感器覆盖及策略训练域不据此转为已通过。

## 第一阶段历史结果

本节保留先前的实现和局限记录，不能替代上表第二阶段的实际物理结果。

| Robot | 按方案完成的实现/检查 | 当前验证范围与剩余问题 |
| --- | --- | --- |
| [Husky＋UR5](Robot_Husky_UR5/WBC.md) | 真正 OCS2 非完整动力学，9 状态/8 输入；真实 SLQ/MPC 与 USD 相机模型 | CPU 预测域通过；尚未用该 MPC 驱动 Isaac 轮驱物理闭环 |
| [Ridgeback＋UR5](Robot_Ridgeback_UR5/WBC.md) | 真正 OCS2 全向扩展，9 状态/9 输入，包含 `vy_body` | CPU 预测域通过；物理桥未接通，原平面关节底盘不等于轮地动力学 |
| [TIAGo](Robot_TIAGo/WBC.md) | 当前上游 Giskard QP/PIQP 核心组件；单臂差速、torso、7 轴臂的连续相机任务 | 800 步 CPU 运动学验证通过；完整 World/Executor 与物理桥尚未接通 |
| [G1](Robot_G1/WBC.md) | AGILE velocity-height 管 12 个腿部关节；真正 Pink 管 15 个上半身关节；腰 roll/pitch 独立保持 | 6 秒前进＋相机补偿物理短验通过；实际速度有明显增益偏差，尚未完成新组合全方向验收 |
| [Go2＋Piper](Robot_Go2/WBC.md) | 确认原生 WBC 策略与实际 Piper 一致；补齐全外参和倾斜机身的 mixed-frame 转换 | 站立局部相机轨迹通过；移动协调案例位置 RMSE 约 331 mm，未通过 |

两款 OCS2 的动力学解析导数分别在 128 个状态上与独立有限差分一致，最大误差
约 `1.84e-10` / `2.90e-10`。两者都实际运行了上游 `GaussNewtonDDP_MPC/SLQ`，
不是给原 NumPy 控制器改名。其一秒预测域在模型内通过，不能据此替代物理执行。
目前边界使用 relaxed barrier，预测采样点事后检查通过；加速度、刹停、滑移和碰撞约束
仍需接入，不能将既有执行端的限速/加速度处理当作预测模型的一部分。

TIAGo 的两个 4 秒 CPU 任务分别为机械臂/torso 独立运动、差速底盘与臂协调运动。
相机位置 RMSE 约 0.302 / 0.239 mm，角度 RMSE 约 0.000316 / 0.000271 rad；
协调任务的模型底盘移动约 9.13 cm。这些是**运动学核心组件结果**，不代表完整 Giskard
任务图或 Isaac 物理验收。现有稳定轮驱入口保留，不静默切换到尚未接好的新框架。

G1 新组合的物理相机位置 RMSE **3.85 mm**、姿态 RMSE **0.00590 rad**，
FK 对实际原始相机最大位置误差约 0.465 mm。实际活动阶段前进约 **3.79 cm**，
明显小于 `vx=.05 m/s` 命令的理想位移；上半身关节运动范围的合成范数约 0.266 rad。
这是带真实身体运动的相机补偿短验，不能推广为精确速度伺服。
Pink 使用独立求解进程避开 Isaac 内的二进制依赖冲突；物理不等待该进程，
过期结果丢弃，缺新命令时保持上半身目标并给底盘停止命令，不复用旧速度无限执行。

Go2 的站立局部相机短验位置 RMSE **19.1 mm**、姿态 RMSE **0.0115 rad**。
随后移动协调短验在保持世界相机参考时，原生命令的 `x_link0` 从约 0.50 m 降到
**−0.236 m**；这已越过上游配置的正向目标范围。实测位置 RMSE **330.6 mm**，
且转向与所下发正向 yaw 命令相反。不能把“混合坐标转换正确”推导为“策略能执行任意世界相机轨迹”。
该失败原始记录与源代码快照完整保留，连续能力门禁保持关闭。
`Robot_Go2/wbc_backend.py` 新增 `check_upstream_domain` / `camera_command_support`，
直接读取固定上游配置和采样器的外包络。本次 200 条活动命令中有 156 条越界，首次在 2.90 s。
预检对域外明确返回不支持；域内仍为未知，不能认证发布 checkpoint 的真实训练覆盖。
该预检供后续 Agent 适配调用，没有向策略腿部输出叠加修正，也没有裁剪或改写相机目标。

物理误差已通过独立读取保存的完整相机变换重算，原始 RGB 也已独立解码：

* [G1 前进短验独立审计](validation/wbc_g1_camera_probe_audit_final_20261020.json)
* [Go2 站立短验独立审计](validation/wbc_go2_camera_probe_audit_final_20261020.json)
* [Go2 协调失败独立审计](validation/wbc_go2_coordinated_probe_audit_final_20261020.json)

上述均为短诊断，完整连续相机能力、环境约束、统一任务物理入口和避障未认证。
G1 前进短验实时比约 1.000；Go2 协调失败试验同时采集扫描，实时比约 0.779，
时序未通过独立记录，未通过放慢物理时间或降低原门槛处理。

## 验收门槛

`scripts/check_wbc_readiness.py` 同时检查五个目录和原有验收文件的 SHA256。
各适配器用 `evidence_files_sha256` 绑定本轮源码和结果；文件缺失或改变会使检查失败。
正向就绪声明还必须给出已绑定的 `runtime_source_files` 与 `runtime_validation_files`，
历史控制器验收不能作为新后端验证。只读任务探测与整组预检复用同一个证据检查器。
只有以下五项均为真实布尔值 `true` 时，才将某款机器人记为可接入避障 Agent：

1. `backend_runtime_ready`：选定框架与对应机器人适配器实际可运行。
2. `continuous_camera_tracking_verified`：实际物理相机连续跟踪已验证。
3. `environment_constraints_verified`：该控制器的环境约束实际接通并验证。
4. `representation_ingress_verified`：物理循环已接收统一任务并返回实际几何反馈。
5. `sensors_avoidance_verified`：当前传感器配置已支持对应避障场景并通过验证。

协议单测、核心求解器单测、静态配置和旧控制器运动记录分别保留。
它们不会自动使上述任一物理验证项通过。此检查器读取适配器的声明和引用证据，
不能替代独立物理验收。

```bash
cd ros2_ws/jev_agent_wbc
/home/w84384651/miniconda3/envs/env_isaaclab/bin/python scripts/check_wbc_readiness.py
# 本轮收尾：轮式原控制验收＋G1/Go2 明确的基础层；不表示精密或避障已就绪。
/home/w84384651/miniconda3/envs/env_isaaclab/bin/python scripts/check_wbc_readiness.py --require-basic-control-ready
# 仅检查控制运行、实际相机跟踪和统一任务物理入口三项；不表示避障已就绪。
/home/w84384651/miniconda3/envs/env_isaaclab/bin/python scripts/check_wbc_readiness.py --require-control-ready
# 用于下阶段启动门禁；任一项未验证时返回 2。
/home/w84384651/miniconda3/envs/env_isaaclab/bin/python scripts/check_wbc_readiness.py --require-ready
```

## 统一任务与反馈

实现：`jev_agent_wbc/wbc_protocol.py`。复用现有 `jev_geometry/v1` 的 `Pose`、
`Observation`、`Request`，不增加暴露给 Agent 的关节或执行器字段。

* `CameraTrajectory`：世界系中的带时间 6D 相机参考；`start_s` 为绝对仿真时间，
  每个样本的 `t_s` 是相对该起点的时间。平移分段线性插值，旋转沿 SO(3) 最短弧插值。
  `sample_with_twist` 同时提供世界系线速度/角速度前馈。不允许轨迹外推。
  插值不构成速度、加速度或碰撞可行性证明。
* `TimedTask`：相机轨迹、位置/角度容差、绑定的观测 ID，以及可选的现有几何 `Request`。
  身体偏好保持从属关系，不改变相机参考；更改相机参考需要 Agent 明确授权的新任务。
* `Capabilities` / `negotiate`：检查实际运行可用性、观测时效、时钟、轨迹有效期和逐项能力。
  不支持的身体请求明确拒绝，不丢弃后继续执行。`supported=true` 时仍返回 `feasible=null`；
  轨迹能否执行必须由数值后端和物理安全检查判断。
* `measured_feedback`：返回测量时刻的实际相机、位置/姿态误差、两部分几何、数值可行性状态、
  原因和剩余调整能力。未知能力用 `null`，不会用零误差推导避障安全。
  失败原因采用公共状态码；具体关节、数值求解细节留在后端日志。
* `observation_from_measured_bounds`：从采集时刻的实际碰撞体包围盒构造两部分保守几何，
  保留完整机身旋转和高度。适配器负责覆盖所有相关碰撞体；包围盒本身不是安全证书。

Representation 相机轴为 **X 前、Y 左、Z 上**。物理光学相机轴为
**X 右、Y 下、Z 前**。`semantic_camera` / `optical_camera` 显式转换二者，
完整相机外参必须在数值后端内保留。Go2 的原生 mixed-frame 命令是另一种约定，
不能直接接收世界系相机位姿。

`scripts/wbc_task_gateway.py` 可用真实 JSON 检查某个机器人是否接受任务：

```bash
/home/w84384651/miniconda3/envs/env_isaaclab/bin/python scripts/wbc_task_gateway.py \
  --robot-dir Robot_Go2 --task /absolute/task.json \
  --observation /absolute/observation.json --now-s 10.0
```

该入口是**只读协议探测**，不发出运动指令；返回支持也不等于数值可行。
它要求适配器明确声明 `protocol_features`，并且真实 Representation 入口验证通过。
仅安装框架或添加坐标转换函数，探测仍会拒绝执行能力声明。

TIAGo 核心试验采用作者提供线速度的 C1 Hermite/RotationSpline 参考，与公共协议的
分段线性/SO(3) 最短弧参考不同。物理接入时必须保持输入插值和绝对时间；
未经 Agent 授权，不能通过自动平滑或重定时把二者替换为同一条轨迹。

## 当前回归与后续门禁

第二阶段公共任务、实际几何、物理审计、关节限值、完整套件、就绪检查、精确相机采样及
有界高度反馈共 **145 项通过、无跳过**，记录在
[回归 XML](validation/wbc_stage2_shared04_20261020.xml)。新增的完整任务上下文组件另有
[15 项测试](validation/wbc_solver_context01_20261020.xml)和真实子进程的任务切换、迟到回复检查通过；实际执行记录也已独立核查完整任务绑定和时效，不能以这些接口检查替代运动验收。保护早停及公共反馈审计的
[51 项回归](validation/wbc_abort_feedback_audit01_20261020.xml)通过；旧 G1 中缺新鲜命令却报 `accepted` 的记录会被明确拒绝，停车未完成也不会隐藏原生保护失败。

[中期五机器人预检](validation/wbc_readiness_mid_stage2b_20261020.json)确认三款轮式机器人的
源码与物理证据绑定一致，三项控制门禁通过。G1 和 Go2 的三项仍关闭；开发中的 G1 源码
已经变化，须在最终验证后重新绑定。全五控制门禁和环境避障门禁均未放行。
第一阶段的 [213 通过、4 跳过记录](validation/wbc_regression_final_20261020.xml)和
[Giskard 隔离环境 14/14 记录](validation/wbc_giskard_isolated_final_20261020.xml)继续作为历史结果保留。

接下来先完成 G1 重力补偿、速度与停车修正，以及 Go2 前瞻分配、跟踪偏差和停车修正，
再用同一最终配置复验全部方向与非平凡六维相机任务。公共反馈中的约束保持实际采集时刻、
失效期限和明确的未知/失败原因。传感器驱动的环境约束和覆盖验证仍是之后开启避障 Agent 的前提。

## 既有感知与物理边界

本轮不改变原关节范围、传感器盲区记录或历史物理证据。
Husky 与 TIAGo 的相机/LiDAR 全姿态限制见 [MOBILE_MANIPULATORS.md](MOBILE_MANIPULATORS.md)。
Ridgeback 原验收使用平面关节底盘，轮地接触与垂向动力学不在其通过范围内。
G1/Go2 的行走策略与训练命令空间不能推导为任意身体部位可独立调整。
环境约束、传感器可见性、连续物理执行和 Agent 语义许可需要分别验证。

新物理执行保持 `pace=1`；实际耗时与相机/稳定性结果分别记录。
未交付视频，本轮原始传感器图像仅作为测量证据。
