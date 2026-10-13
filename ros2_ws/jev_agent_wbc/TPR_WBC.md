# 轨迹保持的预测构型避障（TPR-WBC）

启用 WBC 避障时，`ExecutionWBC` 现在执行 TPR-WBC。Agent 负责修改 task-space trajectory；WBC 只分配差速底盘与六轴机械臂的运动。收到的相机位置、朝向、时钟和结束时间不因障碍而被 WBC 改写。旧的提前撤退、底盘逃逸朝向、障碍相关相机权重退让，以及失败后调用旧避障器的路径均不在默认执行链路中。

## 控制结构

状态为 `[x,y,yaw,q1..q6]`，输入为 `[v,omega,dq1..dq6]`，通过差速运动映射消去不可执行的横向速度。八个瞬时输入承载六维相机任务，一般只有两个自运动自由度。相机不被障碍物直接碰到，并不意味着其余机体一定能在相机固定的条件下避开任意障碍。

高速层在端点速度上优化，对应加速度 `(u_next-u_measured)/dt`。相机任务使用 received reference 的位置、旋转、速度、数值微分前馈加速度，以及实测速度、`Jdot*u` 补偿。ROS/PhysX 还补偿实测底盘的滚转、俯仰和横向扰动。

1. 硬约束包括关节位置与停止距离、速度/加速度、轮速/轮加速度、任务截止时刻停止包络、外部逐 primitive 的相对运动高阶屏障约束，以及非相邻 link 自碰撞约束。
2. 若相机任务可行，直接施加六维等式，用未阻尼 SVD 消元求解。构型偏好再大也不能改变该输出。
3. 若等式不可行，先只最小化相机任务残差，再固定第一级的六维输出优化构型。此时显式记为 `degraded`；它不是正常跟踪成功。
4. 返回命令之前检查原始约束残差，并做当前控制区间的外部障碍扫掠和自碰撞检查。指令不再经过额外逃逸、平滑或裁剪。安全集合为空、求解失败或扫掠无法认证时抛出错误，不宣称停车必然安全。

相机平移与旋转在不可行模式下分别使用权重 1 和 0.35；此权重仅定义不可行时的最小即时任务损失，不允许正常模式用姿态偏好交换相机误差。数值硬约束容差为 `1e-6`，不是物理安全裕度。

## 构型预测与补偿余量

默认预测窗 1.5 s、6 个区间、0.4 s 更新周期，最多 4 次 SQP 迭代。每个节点重新计算差速运动学、相机雅可比、障碍位置、距离梯度和屏障约束。目标是在相机任务条件下保持运动连续、补充关节停止余量、达到有限的避碰余量；不无限追求离障碍更远。

预测节点约束相机六维位姿，包含输入变化、轮速、关节范围、分段线性停止距离上包络、外部碰撞与自碰撞。接近关节边界时，软代价同时考虑 0.2 s 反应距离和制动距离，并寻求 0.2 rad 储备。外部距离偏好达到安全边界外 40 mm 后停止奖励。预测解经过非线性重新积分和区间扫掠检查才成为次级速度偏好；高速 QP 仍独立保护当前任务。

ROS 默认使用单进程异步预测器：最多一个在途请求，拒绝参考版本不符、超过 0.6 s 或状态偏离过大的结果。高速 QP 不等待 SQP。同步模式供确定性离线 rollout 使用。异步耗时分位数不是完整 ROS 回调或实体机器人实时性证明。

`rho` 由单个 LP 联合计算所有活跃安全行在相机等式和执行能力约束下的最小归一化余量。它不是逐 link 可行性的简单相加，也不是全局路径存在性证明。报告包含 `camera_task_feasible`、`camera_actuator_feasible`、`rho`、参考标识、涉及 link/障碍、首次预测冲突时刻（可检测时）、关节制动余量、观测年龄和速度预测误差。

`warning` 表示当前相机任务仍可行，但预测失败、余量小或关节储备低；`degraded` 表示当前无法同时实现相机任务与安全约束。Engine 把这些报告送入现有 Agent 反馈，并记录事件。候选激活后的任务降级会阻止候选通过；评估的 `received_task_preserved` 和 `within_budget_success` 不将降级算作正常成功。激活之前，候选预测严格沿用当前父参考，避免未安装候选提前改变共同执行前缀。

## 测量、模型与适用边界

- ROS 必须收到关节位置和速度、底盘里程计位姿和 twist。默认时间戳偏差不超过 50 ms，数据年龄不超过 200 ms，缺失或失效时不发布新的认证命令。`state_stamp_skew_s`、`state_timeout_s` 可配置。外部执行器 watchdog 仍是硬件集成要求。
- 命令约束以历史命令为起点，实测约束另以当前速度为起点；物理运行可分别配置保守工作限幅与真实硬件限幅，二者都必须满足。绝不把实际速度重置成命令来掩盖跟踪误差。
- `velocity_response_time_s` 是可选的八维一阶速度响应标定，默认全零（理想端点速度模型）。非零时解析映射命令 ramp 到预测实际速度，并检查整个响应的保守扫掠包络。数学单元测试通过不等于动力学辨识完成；本轮底盘低速摩擦/迟滞尚不能由一个时间常数可靠解释，因此未将试验时间常数设为默认。
- 外部碰撞沿用 URDF primitive 与观测 surfel。动态障碍采用当前位置/速度外推和半径不确定性增长；默认增长率 0.035 m/s，可由运行时观测模型明确提供。不是未来真值接口。
- 自碰撞使用包围胶囊/球，跳过同一刚体、相邻 body 及 neck/link2 的安装重叠。它对纳入的 primitive 对保守，不是全机器人网格接触认证。
- 当前 QP 可行、预测窗检查通过、长期递归可行是不同结论。本实现只报告前两项；线性化、非完整运动、碰撞特征切换、传感延迟及真实伺服误差都可能造成失败。

## 固定相机、持续飞行障碍演示

运行目录必须不存在。以下均从仓库根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 \
  ros2_ws/jev_agent_wbc/scripts/demo_tpr_wbc.py --duration 72 --period 24 \
  --output /tmp/tpr_fixed_camera_72s

OPENBLAS_NUM_THREADS=1 /usr/bin/python3 \
  ros2_ws/jev_agent_wbc/scripts/render_tpr_demo.py /tmp/tpr_fixed_camera_72s
```

半径 70 mm 的球在机身附近连续飞行，相机 received position/rotation 全程恒定。底盘和关节以受限加速度积分更新，不使用 IK 投影，也不直接写入固定相机位姿。WBC 只收到当前障碍状态。夹具用已知加速度范围标定预测误差包络；真实飞行曲线只在环境和独立评估中使用。冻结机体的反事实间距验证该障碍确实会与原构型相交。

异步验证加 `--async-preview --pace 1`；两倍速度压力测试用 `--duration 36 --period 12`。输出包括源文件指纹与配置、全程 telemetry、summary、视频和预览图。通过阈值为：完整执行、最小外部间距 ≥25 mm、相机位置误差 <1 mm、角度误差 <0.002 rad、零任务降级、所有命令限幅满足。

PhysX 版本使用实际关节状态、刚体相机位姿和接触回调，原始约束以 1 kHz 独立审计；默认 2 ms 因果速度估计只作用于控制输入，不过滤审计数据。当前球体检测为理想跟踪器，此演示不是 LiDAR 感知实验：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES /usr/bin/python3 \
  ros2_ws/jev_agent_wbc/scripts/jvs_isaac_bootstrap.py \
  --isaac-python /home/w84384651/miniconda3/envs/env_isaaclab/bin/python \
  --runner ros2_ws/jev_agent_wbc/scripts/demo_tpr_isaac.py -- \
  --output /tmp/tpr_physx_48s --duration 48 --period 48 --record \
  --arm-drive-stiffness 1800 --arm-drive-damping 400 \
  --wheel-drive-damping 1000 --servo-frequency 20 \
  --velocity-window-ms 2 --solver-velocity-iterations 64

OPENBLAS_NUM_THREADS=1 /usr/bin/python3 \
  ros2_ws/jev_agent_wbc/scripts/render_tpr_demo.py /tmp/tpr_physx_48s
```

物理通过判据额外要求零接触和原始物理约束审计通过。运动学通过不能替代物理通过；最终数据、失败迭代与限制见 `results/tpr_wbc_20261013/REPORT.txt`。

物理视频修正：早期录像只向 PhysX 提交 kinematic target，未同步用于绘图的 USD 位姿，造成物理球移动而画面中的球静止。现在独立显示球只使用碰撞球的 PhysX 实测位姿，不向物理碰撞体写回绘图变换；每个物理子步验证实际路径，每帧用 RGB 橙色球心与相机投影交叉核对。`audit_tpr_video.py <运行目录>` 输出 `video_motion_audit.json`，`render_tpr_demo.py` 在画面运动与日志不符时拒绝导出。旧 `physx_48s` / `physx_refined_48s` 视频作废为动态避障演示，替代文件为 `physx_render_corrected_48s/demo.mp4`。

## 验证

```bash
PYTHONPATH=ros2_ws/jev_agent_wbc OPENBLAS_NUM_THREADS=1 /usr/bin/python3 \
  -m pytest -q ros2_ws/jev_agent_wbc/test --disable-warnings
```

新测试覆盖严格任务优先级、动态障碍下底盘与机械臂共同运动、显式降级、禁止旧回退、实际/命令速度不一致、关节与轮约束、异步结果失效、自碰撞梯度、预测快照独立性、障碍模型误差包络以及一阶响应映射。

本轮 123 项测试通过，ROS 包构建成功。72 秒三圈运动学演示最大相机位置误差 0.004925 mm、最小间距 44.479 mm、零任务降级；两倍障碍速度与异步版本也通过。修正渲染后的 48 秒物理演示最大位置误差 0.441856 mm、朝向误差 0.097310°、零接触，但有 16 个任务降级样本且原始加速度审计失败。577 帧均通过球心投影检查，最大误差 0.755 像素。原有 ROS MoveIn 压力场景和 4 秒候选激活时限集成测试尚未通过，不能将这些结果表述为完整系统或硬件安全验证。
