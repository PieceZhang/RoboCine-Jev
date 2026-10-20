# 独立移动机械臂控制与验收

本目录为 `Robot_Husky_UR5` 和单臂差速 `Robot_TIAGo` 提供共享实现，机器人模型、
关节、传感器和来源清单仍分别保存在各自 `Robot_*` 目录。无需调用 Jev 或现有规划算法。
Isaac Sim 使用本机已有的 5.1 环境；没有升级其他机器人的运行环境。

`run.py` 在 100 Hz PhysX 中通过真实车轮速度驱动移动底盘，机械臂使用有力矩上限的
关节驱动。根节点仅在初始化时安放到地面；运行期间不重写底盘或连杆位姿。
Husky 的滑移转向存在轮地接触误差。TIAGo 官方描述的固定脚轮使用显式被动支撑近似，
具体材料与转换记录在机器人目录。两个底盘的横向目标均通过转向后行进完成。

`control.py` 的输入为实测关节状态、实测世界坐标 Jacobian、Camera 光学中心
6D 位姿误差和参考速度。求解变量为 `[v, yaw_rate, arm_and_torso_joint_speeds...]`，
UR5 为 8 维，TIAGo 为 10 维；不添加虚假的底盘横移自由度。
Jacobian 包含相机相对实际末端的完整安装偏移。QP 同时约束位置、速度、加速度和轮速；
无法求解时拒绝发送候选并停车。它是局部运动学 WBC，没有预测控制或完整碰撞路径规划。
不能将其运动验收当成闭环避障或连续构型空间安全认证。

从仓库根目录运行（每次使用新的输出目录）：

```bash
ros2_ws/jev_agent_wbc/Robot_Husky_UR5/run.sh \
  --probe --sensor-audit --sensor-rgb --output /tmp/husky_probe_new
ros2_ws/jev_agent_wbc/Robot_TIAGo/run.sh \
  --sensor-audit --sensor-rgb --output /tmp/tiago_motion_new
```

去掉 `--probe` 执行完整的前后、左右、对角目标、独立机械臂和 Camera 全身轨迹测试。
`--trajectory` 接受世界光学坐标下有时间参数的 JSON 数组：

```json
[
  {"t_s": 0.0, "position_m": [0.6, 0.0, 1.0], "quaternion_xyzw": [0, 0, 0, 1]},
  {"t_s": 4.0, "position_m": [0.8, 0.1, 1.1], "quaternion_xyzw": [0, 0, 0, 1]}
]
```

位置单位为米，四元数必须为单位四元数，时间严格递增并从零开始。外部轨迹按绝对
世界坐标执行，起点需与实际相机状态匹配。时间、位置和姿态格式在创建仿真器前检查。
位置线性插值、旋转 SLERP，参考速度由同一曲线计算，不重新计时或调用上游轨迹修改器。

`--wbc-only --initial-heading-rad -2.35619449` 可在指定初始朝向做约 20 秒的
WBC/停车短诊断；朝向仅在物理开始前设置。该模式跳过方向和独立机械臂测试，
输出明确标记为诊断，不能算作完整验收。执行控制器、传感器实现/配置和外部轨迹
均使用运行开始时的冻结副本。接近关节界限时提前保留制动距离，并通过软代价优先
调动底盘与冗余关节；硬位置、速度及加速度约束不变。

传感器来自现有 `robot_sensor_rig.py`：实际 Camera RTX RGB，以及不能穿透本体的
PhysX 最近命中 LiDAR。每款最多一个主扫描器和一个辅助扫描器，采样与量程参数一致。
LiDAR 是瞬时几何扫描，没有模拟真实型号的扫描时序、噪声、强度或材质响应。
原始自命中、未命中和最小量程盲区保留，不伪造成环境点。它没有接入避障规划器。

姿态与环境目标覆盖另行运行：

```bash
/home/w84384651/miniconda3/envs/env_isaaclab/bin/python \
  ros2_ws/jev_agent_wbc/scripts/validate_robot_sensor_coverage.py \
  --robot-dir ros2_ws/jev_agent_wbc/Robot_Husky_UR5 \
  --poses 64 --output /tmp/husky_coverage_new
```

检查原始关节范围的各端点、固定种子的 Sobol 组合构型、机身姿态、实际 RGB 的本体掩码、
相机外壳代理干涉，以及近距离不同高度的障碍物检测。TIAGo 包含躯干的米制端点。
固定基座几何检查与轮地运动检查分别记录。有限采样不能证明所有连续姿态；
机械臂自身碰撞的构型也会保留，不能通过缩小关节范围或隐藏失败图来声称无遮挡。

`acceptance.json` 为物理运动结果，`summary.json` 为有限姿态与传感器结果，
两者相互独立。错误、缺失产物或未完成运行均不能通过父进程验收。
`samples.jsonl` 保存每个物理步的关节、完整底盘/相机变换、参考与误差；
`sensor_samples.jsonl` 保存采集时刻的扫描和坐标变换。`sources/` 与 `provenance.json`
保留运行时源码和参数。`timing_pass` 单独反映实测计时，不放宽运动或传感器阈值。
原始 RGB 为传感器测量证据；本目录不交付未经过 Representation 流程的整机视频。
