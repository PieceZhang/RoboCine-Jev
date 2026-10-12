# Stage 1 Whole-Body Camera Controller

本阶段提供差速底盘与 6DOF 机械臂统一的速度级 constrained QP 跟踪器。
论文背景为 [Li et al., arXiv:2608.07005](https://arxiv.org/html/2608.07005v1)，
其中 III-A 的移动机械臂状态和轮速/轮加速度关系与这里的模型相对应。
本模块是未来规划器的执行和反馈基础，不等同于复现论文的前端搜索、MINCO 或
KC-SVSDF；本阶段也没有 shot-type preference、障碍物约束或碰撞安全声明。

## 数学模型与约束

状态 `s=[x,y,yaw,q1,...,q6]`，控制 `u=[v,omega,dq1,...,dq6]`。

```text
xdot = cos(yaw) v
ydot = sin(yaw) v
yawdot = omega
qdot = u[2:]
wheel_left  = (v - 0.5*b*omega)/r
wheel_right = (v + 0.5*b*omega)/r
```

因此 `-sin(yaw)*xdot + cos(yaw)*ydot = 0` 由变量参数化精确实现，
没有优化后投影，也没有侧向底盘控制通道。
相机空间 Jacobian `J_camera(s)` 为 6×8，包含底盘平移、绕底盘原点的旋转
及机械臂全部六列。误差为世界坐标中的位置差和 `Log(R_target R_current^T)`。

每周期 OSQP 求解严格凸 QP：

```text
min_u  1/2 ||W (J_camera u - (twist_ff + K e))||²
     + 1/2 uᵀ R u + 1/2 lambda ||u-u_previous||²
subject to
    -u_max <= u <= u_max
    -a_max dt <= u-u_previous <= a_max dt
    -wheel_max <= D u <= wheel_max
    -wheel_acc_max dt <= D(u-u_previous) <= wheel_acc_max dt
    q_min <= q_measured+dt*dq <= q_max
    q_min+margin <= q_command+dt*dq <= q_max-margin
    dt*|dq| + dq²/(2*a_joint) <= distance_to_position_limit
```

最后一项预先转成每关节速度上下界，因此实际优化仍然是线性约束 QP。
它为接近限位时的离散制动保留距离，避免只看下一步位置造成冲过限位。
同时约束实测关节位置与积分位置目标，避免位置伺服误差累积使目标越界。
积分命令路径内缩 0.01 rad，实测路径使用 URDF 硬限位；这段距离用于容纳伺服跟踪误差，
避免将相同内缩边界再施加到实测路径而耗尽预留的制动余量。
相机任务是软目标；执行器约束不加 slack。不可达轨迹会产生跟踪误差，而不会
通过突破执行器限制追赶。求解失败记录 fault 并尝试另一个最小速度制动 QP；
若硬约束已不可行，则标记 emergency stop，明确不保证该故障动作的加速度约束。

默认限制：

| 项目 | 限值 |
|---|---|
| 轮半径 / 轮距 | 0.07 m / 0.21 m |
| 轮速 / 轮加速度 | 15 rad/s / 20 rad/s² |
| base v / omega | 0.5 m/s / 1.2 rad/s |
| base a / alpha | 0.5 m/s² / 1.5 rad/s² |
| joint 速度 | [1, 1, 1, 1.2, 1.2, 1.5] rad/s |
| joint 加速度 | 全部 2 rad/s² |
| joint 位置 | URDF 限位，内缩 0.01 rad |

轮速和关节位置/速度来自 URDF；轮和关节加速度是明确设置的 Stage 1 设计限制，
URDF 没有给出这两项。base 采用比原控制器更保守的运行限制。
这是速度级 WBC，QP 约束保证的是离散命令；有限力矩伺服、接触冲量与物理求解误差
是否满足相同物理约束，必须由独立 Isaac 实测确认，不能由 QP 可行性代替。

## 安装与启动

ROS 使用系统 Python，Isaac 使用 env_isaaclab 的 Python。两者都需要 numpy、scipy、OSQP。
系统 Python 的 OSQP 安装示例：

```bash
/usr/bin/python3 -m pip install --user 'osqp>=0.6.7,<2'
source /opt/ros/humble/setup.bash
cd /home/w84384651/Documents/RoboCine-Yuelin/ros2_ws
colcon build --packages-select jvs_mock --symlink-install
source install/setup.bash
# 已阅读并接受 NVIDIA EULA 后设置：
export OMNI_KIT_ACCEPT_EULA=yes
ros2 launch jvs_mock jvs_wbc_isaac.launch.py headless:=true \
  trajectory_file:=$PWD/jvs_mock/trajectories/wbc_coordinated.json \
  output_directory:=/tmp/jvs_wbc_run
```

本次开发系统 Python 依赖隔离安装于 `/tmp/jvs_wbc_deps`，复用该安装时需将它加入
`PYTHONPATH`，与 ROS 的已有 `PYTHONPATH` 一起保留。环境清理后请重新安装依赖。

新 launch 基于 `jvs_mock_isaac.launch.py`，关闭旧 command adapter 和旧轨迹 follower，
由 WBC 唯一下发 `/isaac_joint_commands`。底盘使用轮速伺服，机械臂使用积分位置目标
及 QP 关节速度前馈，均受原 URDF effort 上限约束。
普通 Isaac launch 的旧适配器仍默认开启。
WBC launch 默认使用 PhysX PGS，原普通 Isaac launch 保持 TGS。
PGS 在此模型上显著降低实测速度抖动；驱动增益、URDF 惯量、力矩和约束限值保持原值。
可用 `solver_type:=TGS` 对照。PGS 本身不代表已满足逐物理步加速度限制。

输入：`/joint_states`、`/base_controller/odom` 和 `trajectory_file`。
输出：`/isaac_joint_commands`、`/wbc/diagnostics`、`telemetry.csv`、`summary.json`。
日志包括位置/姿态误差、base v/omega、全部 q/dq、轮速、QP 残差、命令/实测超限、
求解耗时、横向速度和运行阶段。ROS 相机姿态从实测全姿态 odometry 和实测 q 做 FK；
独立 Isaac 验收使用 camera 刚体姿态，并额外记录 FK 与刚体姿态的一致性误差。

## 轨迹契约

兼容现有 `jvs_relative_task_space_trajectory/v1` JSON：每项为
`time_from_start` 与 `[x,y,z,roll,pitch,yaw]`，单位 s/m/rad。
首样本必须为零相对位姿；时间严格递增。
`base_link` 在 WBC 中表示启动时捕获的底盘坐标轴，固定于 odom，不随运行中的底盘移动；
`initial_camera_link` 表示启动时的相机坐标轴。
位置使用三次样条并令端点速度为零，姿态使用 SO(3) RotationSpline，前馈使用其导数。
轨迹端点后保持目标，2 s 后通过约束制动 QP 停止。
用户轨迹应具有静止起终段以使姿态导数与保持阶段连续；示例已包含这些段。

```bash
PYTHONPATH=/tmp/jvs_wbc_deps /usr/bin/python3 \
  jvs_mock/scripts/generate_wbc_trajectories.py
```

生成正向协同 24 s、反向协同 28 s、快速运动 8 s、不可按时完成的限幅压力测试 4 s。
参考位姿来自一条可实现的差速底盘弧线及六关节平滑运动的 FK，跟踪时不提供这些
底盘/关节参考，控制器只能看到 6D camera trajectory，自行统一分配运动。
压力测试用于验证饱和情况下的约束，不应要求其满足普通轨迹的跟踪误差门槛。

## Isaac 验证方法

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
OMNI_KIT_ACCEPT_EULA=yes /usr/bin/python3 jvs_mock/scripts/jvs_isaac_bootstrap.py \
  --isaac-python /home/w84384651/miniconda3/envs/env_isaaclab/bin/python \
  --runner jvs_mock/scripts/test_wbc_isaac.py -- \
  --xacro jvs_mock/urdf/jvs_mock.urdf.xacro --headless true --enable-lidar false \
  --solver-type PGS \
  --trajectory jvs_mock/trajectories/wbc_coordinated.json \
  --output /tmp/wbc_coordinated
/usr/bin/python3 jvs_mock/scripts/plot_wbc_results.py /tmp/wbc_coordinated
```

Isaac Sim 5.1、PhysX、真实 URDF 刚体/地面接触，physics=1000 Hz、QP=50 Hz。
测试器使用 `sim.step(render=False)` 逐物理步推进，单独 `sim.render()`，检查实际
仿真时钟与控制时间一致。禁止将 `step(render=True)` 当成单个物理步。
`physics.csv.gz` 保存每 1 ms 的原始物理速度和超限，未平滑、未删去运动中的尖峰。
启动后的 2 s 地面接触 settling 不计入跟踪区间；之后跟踪和 3 s 停止阶段均被检查。

完整 ROS 链路：

```bash
OMNI_KIT_ACCEPT_EULA=yes /usr/bin/python3 jvs_mock/scripts/test_wbc_ros_isaac.py \
  --trajectory jvs_mock/trajectories/wbc_coordinated.json --output /tmp/wbc_ros
```

程序等待报告后关闭自己启动的进程组，不影响其他用户进程。
Kit 关闭过程可能把独立 runner 的退出码覆盖为 0，因此以 `summary.json` 的
`completed`、`failure`、`stage1_pass` 为准，不能仅凭进程成功退出判断通过。
`test_wbc_suite.py` 在 Isaac 进程外检查报告，返回 0/2 表示普通轨迹验收通过/失败。
ROS 测试也检查报告和启动退出/超时。
最终报告用 `sampled_checks_pass` 表示状态采样检查；只有独立逐物理步审计也通过时，
才允许 `stage1_pass=true`。ROS 节点自身无法证明中间物理步，因此不会单独宣布 Stage 1 通过。

整组复现（输出目录需不存在各测试子目录）：

```bash
OMNI_KIT_ACCEPT_EULA=yes /usr/bin/python3 jvs_mock/scripts/test_wbc_suite.py \
  --isaac-python /home/w84384651/miniconda3/envs/env_isaaclab/bin/python \
  --output /tmp/wbc_stage1_suite --solver-type PGS --criterion stage1 \
  --cases coordinated reverse fast
```

验收门槛在测试前固定：位置 RMS ≤20 mm、最大 ≤50 mm；姿态 RMS ≤0.035 rad、
最大 ≤0.0873 rad；命令残差 ≤1e-6，物理约束数值容差 1e-3；横向速度 ≤0.005 m/s。
同时要求 base 净位移 >0.02 m、至少一个关节运动范围 >0.02 rad，以及整个运行完成。
报告另外显示命令逐步变化、速度曲线和每物理步的约束超限，供检查控制分配和振荡。
有限轨迹测试不能证明任意目标、任意扰动下的连续时间安全。

实际测试结果与 Stage 1 结论见 `results/REPORT.md`。
