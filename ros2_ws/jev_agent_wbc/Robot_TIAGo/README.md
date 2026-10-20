# TIAGo 单臂移动机械臂（PMB2 差速底盘）

该目录使用 PAL Robotics 官方 TIAGo 描述：PMB2 两轮差速底盘、升降躯干、7 自由度单臂及 PAL 平行夹爪。它不通过虚拟平移关节横移；向左、向右或斜向移动时先改变朝向，再由真实轮关节驱动车身。升降躯干与机械臂均进入独立控制及全身协调控制，控制目标为末端相机的 6D 光学位姿。

源代码版本固定在 [SOURCE_LOCK.json](SOURCE_LOCK.json)，机械模型和网格哈希保存在 [ASSET_MANIFEST.json](ASSET_MANIFEST.json)。下载的是 PAL 官方仓库；网格随目录保留，源仓库可通过固定提交恢复。原仓库许可文本保存在 `LICENSES/`。

2026 年 10 月 20 日的完整 Isaac Sim/PhysX 运动验收及独立审计均通过：前后、左右、两条对角线运动，躯干与单臂独立控制，相机 6D 全身协调，以及最后的位姿/关节停车检查。9131 个连续 100 Hz 样本对应 91.31 秒物理仿真、93.58 秒实测运行时间，`timing_pass=true`。全身相机轨迹位置 RMSE 为 **2.04 mm**、最大误差 **5.95 mm**，方向 RMSE 为 **0.00322 rad**；独立机械臂运动时底盘漂移为 **0.086 mm**。这些结果不表示全姿态传感器验收通过，残余问题见下文。

完整结果、独立审计和实际渲染证据位于 `results/acceptance_final/`：`acceptance.json`、`motion_audit.json`、`overview.png`、`camera_rgb.png`、`camera_direction_probe.png`。三张 PNG 均已解码检查，并人工查看总览和原始相机图像。最后 0.5 秒的躯干位移范围为 **1.16 mm**、相机位置范围为 **1.56 mm**、底盘位置范围为 **0.49 mm**；独立重算的底盘/相机旋转范围分别为 **0.00268/0.00404 rad**。冻结的源码、传感器配置、全部采样、最近命中扫描和扁平化 USD 场景随结果保留。可直接查看 [VALIDATION_RESULTS.json](VALIDATION_RESULTS.json) 获取结构化状态。

上述验收使用共享的本地 SLSQP 控制器，保留为独立历史证据。新增 `run_giskard.sh` 的真实 Giskard World/Goals/Statechart/Executor 已在 Isaac 100 Hz 物理闭环完成最终 FULL04：六方向及每段停车、躯干/单臂往返与归位、16 秒原公共相机任务全部通过，四项独立审计通过。12943 个连续样本、129.43 秒物理/129.436 秒墙钟；相机位置 RMSE **10.23mm**、最大 **16.99mm**、方向 RMSE **.00778rad**，所有瞬时样本均在原 .03m/.08rad 范围内，全部 14 DOF 的严格停车通过。当前控制入口、源码绑定及边界见 [WBC.md](WBC.md) 和 [WBC_READINESS.json](WBC_READINESS.json)，结果在 `results/giskard_executor_full04/`。传感器全姿态覆盖和环境避障仍未通过，见下文。

## 启动与复现

运行环境沿用本工作区 Isaac Sim 5.1 的 Python；可设置 `ISAAC_PYTHON` 覆盖默认路径。首次下载需要网络；物理运行和 RTX 渲染需要可访问 NVIDIA GPU。

```bash
cd ros2_ws/jev_agent_wbc/Robot_TIAGo
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/w84384651/miniconda3/envs/env_isaaclab/bin/python}"
/usr/bin/python3 scripts/prepare_assets.py --fetch
./import.sh
./run.sh --output results/new_acceptance --sensor-audit --sensor-rgb
```

输出目录必须不存在，避免覆盖历史记录。`--probe` 仅运行 5 秒物理诊断；完整运行包含前后、左右、两个斜向的车轮驱动移动、机械臂与躯干独立控制、相机全身协调轨迹及最后停车。`--gui` 可显示 Isaac Sim；默认无窗口。物理步长固定 0.01 秒，`pace=1`；传感器开销造成的实际运行滞后单独记录为 `timing_pass`。

Isaac Sim 可直接打开 [assets/tiago_with_sensors.usda](assets/tiago_with_sensors.usda) 检查安装后的 Camera 和两个 LiDAR 坐标节点。该薄层引用冻结的几何/动力学快照，CPU 检查确认光学轴、相对位姿、双 LiDAR 参数和物理 articulation 保持一致，详见 `SENSOR_ASSET_MANIFEST.json`。LiDAR 节点是由 `run.sh` 驱动查询的传感器坐标定义；单独打开 USD 不会自动输出点云，也没有把它标称为 RTX LiDAR。设置机器人初始姿态和实际扫描仍通过运行器完成。

可向共享运行器传入 `--trajectory trajectory.json`，其中每条样本包含 `t_s`、`position_m` 和 `quaternion_xyzw`，表示世界坐标下的相机光学位姿。位置、方向以及控制输入使用真实车身和末端刚体测量。

```bash
/usr/bin/python3 scripts/preflight.py
"$ISAAC_PYTHON" scripts/validate_kinematics.py
"$ISAAC_PYTHON" ../scripts/mobile_manipulator/audit_motion.py results/new_acceptance
"$ISAAC_PYTHON" ../scripts/validate_robot_sensor_coverage.py --robot-dir . --output results/new_coverage --poses 64
```

传感器验证器的实际参数请使用 `--help` 查看。`CPU_PREFLIGHT.json` 和 `KINEMATIC_PRESERVATION.json` 只证明结构、来源、约束和几何保持；它们不能代替 Isaac Sim 物理、渲染或避障验收。

## 必要的 Isaac Sim 适配

原始展开文件保留为 `urdf/tiago.urdf`，可执行文件为 `urdf/tiago_physics.urdf`。适配保留源模型的所有网格、有效惯量、关节轴、变换、限位和驱动上限：

- 将无质量的 ROS `base_footprint` 参考坐标移出动力学树，以真实 `base_link` 为自由车身根；相关固定子树做精确坐标变换。去掉 6 个没有质量或几何的固定叶节点。273 组配置下，37 个剩余连杆相对车身的变换最大误差为 `4.44e-16`。
- 原模型的 5 个无纹理碰撞 DAE 会触发本机原生转换器崩溃；按原坐标逐三角形转换为 STL，并保留 DAE 原件及转换记录。仅对本地副本的网格文件名用下划线替换连字符，避免非法 USD 标识符。
- 自碰撞保持开启，使用固定版本的 PAL 官方 SRDF 指定允许碰撞对；缺失的原生相机、激光及空坐标对明确列出。没有整体关闭自碰撞。
- PAL 原始脚轮关节为固定关节，碰撞形状为支撑球。使用最小组合模式的 `0.001` 支撑摩擦作为被动脚轮近似；真实轮关节仍通过接触驱动车身。该模型没有验证真实脚轮的旋转、滚动阻力或复杂地面性能。
- 所有受控关节使用力/力矩模式驱动，保留官方最大力及速度。原 URDF 的车轮静摩擦力矩不能直接作为 PhysX 载荷相关摩擦系数，因此省略该不兼容系数，保留阻尼及官方每轮 6 N·m 上限。具体改动逐项保存在 `assets/usd/tiago.finalize.json`。

## 传感器与验收边界

末端相机固定在真实 `gripper_link`，安装偏移为 `[0.16, -0.04, -0.24]` 米；光学前向沿夹爪的局部 `-Z`，上向沿 `+X`。车身安装一台主 LiDAR 和一台辅助 LiDAR，分别位于 `base_link` 的 `[0.29, 0.25, 0.32]` 与 `[-0.29, -0.25, 0.32]` 米。两台共同继承 180 个水平样本、`-84°…84°` 的 2° 俯仰间隔及 `0.05…10` 米量程，避免配置漂移。`sensors.json` 是运行配置，独立姿态验证同时保存真实 RGB、最近命中射线、遮挡掩码和目标检测原始数据。

最终配置已做 81 个有限姿态的实际 RTX/PhysX 审计，以及 27 个邻近安装偏移的比较。`results/coverage_best/summary.json` 明确记录：目标检测为 `29115/29160`（99.846%），仍有 45 次漏检；原机器人无非相邻本体重叠的 55 个姿态中，4 个仍出现相机遮挡，其中 1 个有外壳干涉。因此，该安装方案是经测量改进的原型，**没有通过全姿态传感器或避障充分性验收**。不能把检测率接近 100% 当成安全通过。

残余相机问题及真实图像保存在 `results/coverage_best/unsafe_camera_postures.json`，其构型清单另存为 [SENSOR_UNSAFE_POSTURES.json](SENSOR_UNSAFE_POSTURES.json)：`arm_5_joint_low`（夹爪朝回机身，相机进入底盘/躯干）、`sobol_007`、`sobol_030`、`sobol_048`。保持原关节范围，没有删除这些失败配置。`sensors_initial.json` 和 `sensors_candidate.json` 保留改进过程，`sensors_best.json` 与最终配置相同。

完整物理验收与独立运动审计需要同时通过。有限姿态检测只证明已采样状态；不能据此声称连续全工作空间无盲区，也不能将姿态摆放审计视为可安全到达该姿态的轨迹。相机和 LiDAR 是无质量安装原型，尚未建立真实支架、线缆和电源的碰撞模型。本目录的局部全身控制器没有接入 Jev/Agent、环境路径规划器或闭环避障决策；环境观测充分性、可用安全工作区和实际避障必须分别报告。
