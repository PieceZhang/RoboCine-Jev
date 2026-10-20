# TIAGo：真实 Giskard World/Executor 与 Isaac 物理接入

当前上游的 `WorldWithDiffDriveRobot → CartesianPose / ActuatorRateTask / SelfCollisionAvoidance → MotionStatechart → Executor → QPController` 已实际接通 Isaac 100 Hz 物理闭环。最终 `results/giskard_executor_full04` 的六方向、独立躯干/单臂往返与归位、16 秒公共相机任务及每段固定 5 秒严格停车全部通过，根级 physical、suite、joint_limits、render 四项独立审计全部通过。完整任务链和独立 Giskard/FCL 安全投影分别记录，没有以旧 SLSQP 或单独 PIQP 调用冒充完整框架。

12943 个连续实测样本对应 **129.43 秒物理 / 129.4363 秒墙钟**。相机位置 RMSE **10.232 mm**、最大 **16.986 mm**，姿态 RMSE **.007782 rad**；1600 个运动样本均未超出原瞬时 .03m/.08rad 容差，终点误差 **3.856 mm / .014825 rad**。全部 14 个物理 DOF 的原始位置/速度检查、各段末尾固定完整 .5 秒的严格停车均通过。三个控制门禁为 true；环境约束、全姿态传感器和避障门禁仍为 false，结构化状态见 `WBC_READINESS.json`。

原 `run.sh`、原关节范围、USD、传感器和 `results/acceptance_final` 保持原样。历史 SLSQP 的相机位置 RMSE 2.04 mm、方向 RMSE 0.00322 rad，不能证明新 Giskard 后端通过。本轮独立入口为 `run_giskard.sh`。

## 框架与环境

[当前官方核心](https://github.com/cram2/cognitive_robot_abstract_machine/tree/main/giskardpy)固定在 `7b27424de7768798dc561f447e11c2a8eb86c307`，加载 `giskardpy`、`krrood`、`semantic_digital_twin` 的未修改源码。启动核对真实 Git 提交和被跟踪文件；`big_ball_mode=false`。来源见 `wbc/FRAMEWORK_SOURCE_LOCK.json`，许可证见 `wbc/licenses`。[官方 ROS2 包装](https://github.com/SemRoCo/giskardpy_ros/tree/ros2-jazzy-main)面向 Jazzy，其双臂示例不等同于本机 Humble 单臂；本轮直接接核心，没有宣称 ROS2 包装运行完成。

独立 Python3.11 环境 `wbc/.venv` 已补齐 `probabilistic-model==26.7.0` 等完整 World 依赖，`pip check` 无冲突，清单为 `wbc/requirements.world.lock.txt`。历史组件清单 `requirements.lock.txt` 保留。宿主 Isaac NumPy1.26 和隔离 Worker NumPy2.4 通过普通 Python 数值协议通信，没有升级 Isaac/ROS 环境。

## 模型、控制与硬边界

| 项目 | 实现与边界 |
|---|---|
| 执行坐标 | 差速前向/偏航 + `torso_lift_joint` + `arm_1_joint`…`arm_7_joint`，10 维，无虚拟横移 |
| World 定位 | 被动定位写入实测完整底盘 SE(3)，活动定位自由度为 0 |
| 非任务关节 | 控制 World 冻结轮/头/夹爪到物理保持值；真实物理 14 DOF 仍逐一测量，原增益和 effort 保持 |
| 原限制 | 原 URDF 位置/速度/effort 保留；issued 前向≤0.22 m/s、偏航≤0.35 rad/s、轮速≤8 rad/s |
| 实际下发安全 | 原 issued 加速度 `[0.45,0.65,1,…,1]`、原位置余量和离散制动距离由真实上游 QP 硬投影与独立 guard 检查 |
| 原始实测速度 | 全 14 DOF 原位置/速度、底盘速度及 8 个受控关节制动余量独立检查，不以命令替代测量 |
| 自碰撞 | 上游真实 SelfCollisionAvoidance + 官方允许对；外部 FCL 使用真实 USD 28 个碰撞形状、实测刚体 SE(3) 和物理雅可比构造硬距离速度屏障 |
| 不可控碰撞对 | 非允许对先检查真实距离/穿透，再判断能否构造 barrier；零相对雅可比不会跳过碰撞检查 |

控制副本省略 visual，将 ASCII STL 无损变为二进制并复制全部执行碰撞网格；物理模型未改。每次输出冻结 `control_world.urdf`、执行网格、原模型/相机配置 SHA、后端与共享协议源码。物理前准备模型再与 live Stage 逐项对照，28 个形状此前最大差异约 `1.12e-16 m`。

默认仍为完整 jerk 级上游 QP：100 Hz、horizon 50、`braking_time=0.05 s`。URDF 未声明 acceleration/jerk，没有将附加 issued 策略伪装为硬件 jerk 能力。native 导数预测状态明确为**上一实际下发速度 + 零加速度估计**；q、底盘和光学相机全姿态保持实测。原始 qd 独立记录/守卫，不是预测导数状态。原实际 issued 加速度、位置制动和 FCL 硬约束未改变；该配置不等于动力学或力矩能力认证。

完整测试候选另在真实 native QP 内增加按限位距离变化的 velocity cost：机械臂 `.002+.5*pressure`、底盘前向/偏航 `.06/.03`；pressure 只在 `min(.25,.2*range)` 范围内二次增加，不缩原硬范围。上游内置 `JointPositionList` 以 `.003` 权重仅在 camera 模式提供原 home 软偏好。完整新测试在物理前冻结 801 点原 home+A*sin 的 8 秒关节往返参考，再单独运行预定 5 秒实测 home P 回归，通过原 8 关节 home 容差后，才进入预定 5 秒真正零指令停车；保留已解释的重力补偿 drive target 偏置，无目标跳变。之后才授权/冻结新 camera Task。每段方向、独立单臂和最终相机的停车均预先固定 5 秒，只检查其末尾完整 0.5 秒，没有后验窗口选择。

停车候选仅改变实际车轮驱动的静态保持：明确停车、native 轮速为零且连续 3 个实测样本轮速低于 .08 rad/s 后，一次锁存实际轮编码器位置。驻车 kp300、原 kd30、原 6Nm drive cap，无外加关节力；移动或安全避让的任何非零 native 命令立即释放为 kp0 速度驱动。锁存时刻、实际 q/qd、释放原因和原 effort cap 读回另存，不移动刚体或关节状态来伪造停车。

[PhysX 官方定义](https://nvidia-omniverse.github.io/PhysX/physx/5.1.1/docs/Articulations.html)的 dense Jacobian 在线速度部分参考 link COM。新候选将完整 world `[linear;angular]` Jacobian 平移到 actor origin，再用于 FCL 最近点和光学点；之前逐体/批量相等检查无法独立发现该参考点错误。新增原 URDF 多姿态中央差分 CPU 检查；启动时在第一物理步前，对 3 种 torso/arm 与 base 全姿态组合、全部测量刚体及 10 列，用实际 PhysX actor transforms 中央差分和原 URDF 解析 Jacobian 双重核验，并恢复/核对真实初始 q/T。此项初始状态检查不计作运动或动力学验收。

## 公共任务、时间与反馈

物理入口使用共享 `LiveTaskSession` 接收冻结的公共 `TimedTask`，保存原任务、接纳时实际 `Observation`、绑定 ID 和绝对仿真时钟。公共轨迹的分段线性位置与最短 SO(3) 插值保持不变，没有改成旧组件的 Hermite/RotationSpline，没有重定时。每步更新原始 optical 目标到真实上游 `CartesianPose`；当前没有额外的 native 速度前馈任务。

两部件几何来自实际 Collision GPrim 的 body-local 几何与实测全 SE(3)。躯干上方连杆归 `camera_manipulator`，其余为 `mobile_platform`，每个真实 collider 都需覆盖。反馈含相机位置/姿态误差及 configuration、actuation、self_collision、environment、policy_domain 状态、采样时间和来源。环境缺失为 **unknown**。能力声明仅含 `timed_camera_se3`；要求未实现部件偏好的非空 Request 明确为 **unsupported**，不会静默忽略。

物理 100 Hz、`pace=1`；预热只在物理开始前发生。运行中单包有界异步、命令最大年龄 40 ms；过期或切换阶段按原加速度边界向零制动。求解超过 100 ms 后有限重启，恢复必须有新状态对应的新命令；物理时间不等待求解。

## 实际证据与诊断

`results/giskard_executor_full04` 是当前正式物理证据，源码和原始任务均冻结。六方向实际前进距离为 .3612–.3683m，最大横向误差 .03166m；独立躯干 span .136847m，7 个机械臂关节 span .2404–.2407rad，底盘漂移 4.73e-7m；归位后的 8 个实际关节均满足原 .002m/.01rad home 容差。所有 8 段固定 5 秒停车均通过：最终相机停车的 base/camera 位置 span .775/1.891μm、旋转 span 3.08e-6/3.45e-6rad，末帧实际底盘线/角速度 .0006193m/s / .0005127rad/s。

物理与观测为 100Hz；有效新鲜命令租约占 99.521%，12397 次不同求解响应对应 **95.782Hz**，不能把租约比例称为每步均有新求解。Worker 中位 6.660ms/P99 11.534ms；本次没有 Worker 重启，不据此证明物理故障恢复。实际 raw camera、光轴红色探针、overview 在计时结束后渲染并解码，`camera_render_check.json` 标记 stamp129.43、physics_advanced=false。它们是传感器测量/诊断图像，未制作闭环交付视频，也不代替全姿态覆盖检查。

独立审计位于 `../validation/wbc_tiago_full04_{physical,suite,joint_limits,render}_independent_20261020.json`，前三项对应冻结 auditor 源码也保留。审计重新检查原 Task 插值/绝对时钟/观测绑定与几何、所有 14 DOF、每个方向/独臂的运动与严格停车、原物理限位及实时性，没有只复用运行器的 passed 字段。结构化 SHA 绑定包含这些审计、实际执行源码和原始任务/采样/反馈。

`results/giskard_executor_probe05/acceptance.json` 实际通过：前向 215.8 mm，7 秒物理/7.072 秒墙钟，99.57% 新鲜命令；末 0.5 秒 base/camera 位置 span 为 0.181/0.265 mm、旋转 span 为 0.000287/0.000293 rad，最大轮关节 span 为 0.00219 rad，全部 14 DOF 停车通过。它只是前向短测，不能替代完整运动和相机验收。

`results/giskard_executor_camera02/acceptance.json` 的完整 16 秒公共相机任务、原严格 14DOF 停车和 pace1 全通过：位置 RMSE 8.333 mm/max11.175 mm，方向 RMSE .00743 rad，21 秒物理/21.197 秒墙钟。批量读回与原逐体 PhysX 位姿、原 body-J 公式核验误差均为零；同 tick 缓存使每个物理步只采集一次。根独立审计亦通过。99.86% 表示在有效租约内的命令比例；独立求解响应为 1918/21=91.33 Hz，物理与观测仍为 100 Hz。这轮未含后续 native 软姿态候选，保留为独立版本证据。

候选在 `wbc/results/full_posture_sw_camera_cpu02` 和 `full_posture_forward_camera_cpu01` 的完整 native CPU fixture 均通过，位置 RMSE 8.518/8.228 mm、方向 .00672/.00749 rad，原线性/SO3 时间不变。此前无软偏好的 SW fixture 虽被硬制动约束保护，跟踪仍失败，负例保留于 `full_issued_velocity_sw_camera_cpu02`。

`wbc/results/native_tests_14.xml`：27 项 CPU 测试通过，含真实完整 World/Executor、原限位、回弹和命令状态分离、真实 FCL 分离梯度/非允许穿透/零雅可比检查、全部 14 DOF 速度检查，以及独立多姿态 COM/actor Jacobian 差分和驻车模式进入/释放。

`results/giskard_executor_full01`：新 native 软偏好版本在 54.24 秒右方向固定 5 秒停车末窗失败；两个车轮角 span 为 .032019/.014829 rad，超过原 .01 rad。其余 12 DOF、base/camera 位置/旋转及最终底盘全轴线/角速度均通过。该窗 native proposal 和 issued vw 均为零，轮驱 kp0，实际轮子存在接触抖动与爬动；新候选补驻车位置恢复。未执行独立单臂和公共相机，完整验收失败，原始样本与根级负向审计保留。

`results/giskard_executor_right01`：新驻车/COM 修正版本的原 right 转向、3 秒 drive 和固定 5 秒停车短测实际通过。位移 .362264m，轮末窗 span 3.10e-5/2.95e-5rad，base/camera 位置 span 3.34/9.55μm；16.6 秒物理/16.6064 秒墙钟。实际 3 姿态×25 刚体×10 列校验通过：原 URDF Jacobian 最大差 5.73e-7，真实 PhysX 中央差分最大线/角误差 6.49e-5/1.10e-4，初始状态恢复核验通过。该短测不计作完整就绪。

`results/giskard_executor_full02`：六方向和六段严格停车全部通过。105.47 秒独立臂归位末窗的位姿/14 关节 span 全通过，但末帧底盘线/角速度 .02105m/s / .06297rad/s 超过原 .02/.03；当时仍在用实测 home P 持续修正。原失败保留。新模式明确区分 home 回归活动和零指令停车，不选择有利时间窗。

`results/giskard_executor_arm01`：相同 8 秒原关节轨迹 + 5 秒实测 home 回归 + 5 秒零指令停车短测全通过，20 秒物理/20.0067 秒墙钟。最终 home 误差 torso6.85μm、机械臂最大4.27e-5rad，base drift .555mm；严格停车 base/camera span .206/.671mm、角span .000514/.000918rad，末底盘速度 .001834m/s / .001456rad/s，全部 14 DOF 通过。该短测仍不替代六方向加完整相机的最终验收。

`results/giskard_executor_full03`：六方向、独立臂往返/归位/零指令停车和完整 16 秒公共相机跟踪均通过；相机位置 RMSE11.258mm、姿态 .008697rad，最终原Task误差4.314mm/.01663rad。原固定5秒末窗各位姿/14关节 span 及底盘线速通过，只有原始底盘角速度 .047488rad/s 超过原 .03，完整验收失败。末窗 native proposal≤4.6e-13，安全 PIQP 数值容差留下 torso约8.24e-7m/s 微爬，真实接触/驱动瞬时速度抖动尚需物理诊断；不能用位置差分替代原始速度。

`wbc/results/stop_projection_cpu01`：对51个记录的真实零 native proposal，用原URDF和实际14q/base重建FK，真实PIQP默认投影max8.2394e-7，收紧 eps1e-10/gap1e-12 后max5.59e-12，原硬边界不变，.51秒fixture/.51006秒墙钟。它证明数值爬动消除，不证明真实停车稳定。

物理数值候选参考 [官方TGS受力积分说明](https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/RigidBodyDynamics.html#constraint-solver)：启用每个position子迭代施加重力/外力，并非改变质量或力。`results/giskard_executor_camera03` 的force=True/原64:16短验确实读取True且质量/惯量/摩擦/effort/dt对照未变，但3.30秒wheel-left原始速度8.721rad/s超出原8，独立guard中止，失败保留。`camera04` 恢复 force=False/原64:16 且使用严格 PIQP，消除了全部 8 个 drive target 的末窗微爬，但实际底盘角速度 .046089rad/s 仍超原 .03，不能把求解数值精度改善当作物理稳定修复。最终保持原 force=False/64:16。

`wbc/results/single_floor_static_cpu01.json` 实际原生 USD 检查发现 SDK GroundPlane 同时启用了 `/World/Ground/geom` Mesh 和 `/World/Ground/collisionPlane`，只有 Mesh 绑定预期 .8/.6 材质，analytic plane 未绑定物理材质。新运行只保留原 analytic plane 碰撞并绑定同一个 GroundMaterial，Mesh 仅保留显示；两个地面变换与原材质值逐项不变，机器人模型、allowed pairs、offset、effort 均未触碰。`camera05` 和最终 `full04` 的启动/reset 后 gate 均通过，全部实际停车及 pace 通过。该实际场景修复与结果对应；不能把成功归因于已失败的 force flag，严格 PIQP 本身也未解决 `camera04` 的物理抖动。

`camera05` 为相同单地面配置的 SW 原相机短测，23 秒物理/23.006 秒墙钟，位置 RMSE11.482mm、最大30.253mm；原 RMS+terminal 门槛和严格停车通过，但 9/1600 个瞬时位置样本略超30mm（最长9样本，13.81–13.89s），历史诊断完整保留。最终 FULL04 的逐点超容差计数为零，两轮结果分别记录，没有删除短测峰值或改参考曲线。

`wbc/results/full_issued_velocity_step_stop01/summary.json`：100 Hz 200 步 CPU 阶跃+零请求，2.00006 秒墙钟、99% 新鲜、中位 5.39 ms/P99 7.13 ms，issued 加速度保持原界、末命令归零。此为源 FK fixture，不是物理停车认证。

`wbc/results/full_issued_camera_cpu01/summary.json`：16 秒原时序 linear/SO3 完整 native 相机 fixture，位置 RMSE 9.08 mm、方向 0.00770 rad、99.875% 新鲜，16.00006 秒墙钟。此项采用 earlier issued velocity/差分 acceleration 配置，仅为诊断，不代替最终配置物理证据。

| 保留输出 | 结论 |
|---|---|
| `results/giskard_executor_probe01` | 预热未完成，未执行物理步骤 |
| `results/giskard_executor_probe02` | 实测速度差分注入高阶状态导致零请求机械臂发散，安全 guard 拦截，失败 |
| `results/giskard_executor_probe03` | 单臂稳定但追随躯干速度回弹，制动 guard 拦截，失败 |
| `results/giskard_executor_probe04` | 异步延迟下 issued acceleration 差分形成限环；方向位移和轮关节停车失败，timing 通过 |

`max_derivative=velocity` 也实际尝试：固定上游的 `DegreeOfFreedomDecisionVariables.slots` 硬编码 jerk 列，编译发生 CasADi 越界；保留 `wbc/results/full_velocity_bridge01/worker.log`，没有修改上游伪装支持。

## 复现

在 Robot_TIAGo 目录运行，所有输出必须是新路径：

```bash
wbc/.venv/bin/python wbc/preflight.py --output wbc/results/new_preflight.json
wbc/.venv/bin/python -m pytest -q ../test/test_wbc_tiago_backend.py ../test/test_wbc_tiago_native_runtime.py
./run_giskard.sh --probe --output results/new_giskard_probe
./run_giskard.sh --wbc-only --tgs-force-every-iteration 0 --tgs-velocity-iterations 16 --output results/new_giskard_camera
./run_giskard.sh --direction-probe right --output results/new_right_diagnostic
./run_giskard.sh --arm-only --output results/new_arm_diagnostic
./run_giskard.sh --tgs-force-every-iteration 0 --tgs-velocity-iterations 16 --sensor-rgb --output results/new_giskard_full
bash wbc/bootstrap_world.sh /absolute/path/python3.11 /absolute/path/NEW-venv
```

新环境重建拒绝修改已存在环境。`--backend giskard-core` 明确是组件诊断，默认 `giskard-executor` 为完整框架。`--task` 保持公共观测绑定/新鲜度；`--trajectory` 保留作者原结点与 optical linear/SO3 曲线。

## 剩余门槛

当前 native 配置已经完成有限平地六方向、独立躯干/单臂、原公共相机物理跟踪、逐段严格停车及四项独立审计。全部旧失败、原曲线、关节范围、质量/摩擦/effort 和固定时间窗保留。物理故障注入/恢复、复杂地面和连续扫掠碰撞认证尚未完成。

未接入新鲜环境地图、传感器避障约束或动态障碍试验，`environment_constraints_verified=false`。历史 81 姿态传感器审计仍有 45 次 LiDAR 目标漏检，55 个无本体重叠姿态中有 4 个相机遮挡，`sensors_avoidance_verified=false`。双 LiDAR 参数一致，未修改安装配置或删除失败姿态。
