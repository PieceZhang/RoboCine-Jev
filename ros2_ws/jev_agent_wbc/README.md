# Jev cinematic WBC

独立 ROS 2 Python 包：上游只提交原始相机轨迹、shot phase、主体身份及简短意图；Jev 通过本机 `systemone` 的 `perplexity` 模型在线决定允许的修改，数学规划器选择实际修改量，整机控制器持续检查执行安全。包内已复制原动态 WBC 所需代码、URDF、网格和 Isaac 启动组件，运行不依赖 `jvs_mock`。

这是针对当前 JVS 差速底盘与六轴机械臂的通用运镜原型。通用性指六维相机轨迹和运镜意图，不代表能直接控制任意机器人。当前评估都是单镜头；镜头内部有 establish/develop/resolve 三个阶段，尚未评估多镜头衔接。

本轮结果及未通过项见源码目录下的 [EVALUATION.md](EVALUATION.md)。最终 12 组对照中 11 组执行到预算终点，10 组通过已测安全约束，两种模式各只有 Orbit 满足全部固定几何阶段判据。仍有底盘角加速度超限、一次执行中断和部分视觉退化，整体开发目标尚未完全达到。

## 接口与职责

`scenarios/<case>/task.json` 严格只接受四个字段：

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

上例仅示意字段；合法轨迹至少需要三个时刻，时间从 0 开始、严格递增，阶段连续覆盖原始预算。世界坐标 Z 向上，相机 link 的 X 向前、Y 向左。四元数为 XYZW。不能在此接口增加障碍轨迹、机器人状态或视觉容差。

`runtime_scene.json` 是测试运行器的环境配置：机器人初态、主体初始标定、用于生成传感器数据和独立评分的障碍运动真值。规划器只收到当前点云生成的地图和运动估计，不接收障碍物未来真值。ROS 模式从里程计、关节、TF、双 LiDAR 和主体 PoseStamped 获得运行时信息。

主要代码：

- `contracts.py`：四字段任务、原始轨迹哈希、数值权限窗口、保护属性、过渡范围与收紧后范围的校验。
- `engine.py`：事件触发、异步 Jev、独立数学进程、反馈与恢复、执行前最新状态复核。
- `jev.py`：实际 API 协议与 prompt；Jev 的结构化权限由可审计问答组成。
- `planning.py`：整机困难预测、有效方向探测、有界轨迹候选搜索、全剩余预算预测。
- `reference.py`：始终相对不可变原轨迹构造 C2 参数变化，避免逐轮累积偏离。
- `safety.py`、`dynamic_*`：整机动态避障、差速约束、关节制动界限和连续扫掠证书。
- `evaluation.py`、`scripts/evaluate_sam2.py`：固定评价协议与离线 SAM 2.1 large。

## Jev 问答流程

服务地址默认 `http://127.0.0.1:8000/v1/systemone`，每次请求固定 `model="perplexity"`。不假设此 API 能自由生成 JSON；实际使用它返回的 `choice` 和 `noul`：

1. 根据当前/后续阶段、意图、预测困难、整机余量、障碍估计和执行历史，先判断是否保护主体瞄准、运动方向、水平线，再从相容方向中选择主要修改方向。
2. 必要时选择一个补充方向。保护属性与修改方向矛盾时拒绝该方案并反馈。
3. 对每个选中参数做六轮数值接受性搜索。端点来自机械搜索域；逐次询问“相对原始参考，0 到这个数值之间的全部取值是否符合意图”。视觉容差没有预先写入任务或代码菜单。
4. Jev 决定生效窗口及保留/恢复方式，并对完整数值权限再次确认。恢复事件额外选择 retain/tighten/revoke，复用同一流程。
5. 本地严格校验数值、原始轨迹哈希、时限和保护属性。数学模块验证失败、权限过期或最新状态不适用时，保留安全控制并把原因反馈给下一次事件决策。

数值问题附带该候选边界的三维视角变化和距离平方反比面积尺度估计，并提供运行时主体位置与原始参考姿态；这些是几何提示，不是 SAM 测量或接受阈值。完整权限的数值在最终确认时保持原精度。验证失败反馈保留具体候选、失败原因、净距和关节余量摘要。

可授权参数为 `radial_m`、`lateral_m`、`height_m`、`yaw_rad`、`pitch_rad`、`roll_rad`。它们相对原始参考定义；radial 为主体至原始相机的水平径向，lateral 为原始相机的左向。角度单位 rad，距离单位 m。`bounds` 记录 Jev 明确批准的过渡联合范围，`settled_bounds` 记录平滑过渡结束后的范围。收紧、撤销不会把旧轨迹当作新参考。

数学搜索包含零修改、区间内部值和端点，全部通过同一整机约束模型验证。可行候选首先最小化预测执行相对原始轨迹的位姿误差积分及终点误差；跟踪残差、关节余量、安全余量等只作为次级排序，不能用“修改后目标更容易跟踪”抵消更大的原始参考偏差。该位姿目标是规划代理，独立于四项离线评价指标。允许范围大不要求用满；允许保留也不禁止优化器在范围内提前恢复到零。语义接受概率门槛是模型决策置信度，不是视觉评价阈值。所有请求、原始回答、用量和时延保存在 `jev_api.jsonl`。

全预算验证会保留不同修改幅度及边界候选，避免相同幅度的保留/恢复变体占满名额、漏掉较大幅度的可行解；授权边界的恢复变体也会单独检查。

原始目标需要大幅安全退让时，预测记录会保留实际相机偏差及跟踪残差；零目标偏移不等于零执行误差。方向保护在激活和恢复期间均检查；执行前重新调整激活时刻后，也复核整个剩余目标路径。保留既有权限的恢复决策只需重新确认保护属性、时间和完整方案，不重复搜索已经批准的数值区间。

## 六组短场景

全部为 8 秒、41 个参考关键帧；establish 0–1.2 s、develop 1.2–6.4 s、resolve 6.4–8 s。主体均为带黄色顶部的蓝色圆柱，具体世界位置由运行时提供。每组只有一个移动球体，半径 0.10 m，横向速度约 0.15 m/s，约第 5 秒穿过对应区域。

| 场景 | 原始运镜 | 简短意图 | 障碍设置 |
|---|---|---|---|
| orbit | 绕主体左移约 0.16 rad | 展示主体侧面，保持可辨识的主体构图 | 底盘前侧横穿 |
| movein | 向主体推进 0.28 m | 增强主体强调感，保留接近趋势 | 较靠前的底盘行进区域横穿 |
| truck_left | 向左平移 0.24 m | 展示右侧背景视差，保留左移方向 | 左向运镜的底盘前侧横穿 |
| truck_right | 向右平移 0.24 m | 展示左侧背景视差，保留右移方向 | 对称方向横穿 |
| pedestal_up | 升高 0.065 m | 揭示主体顶部，保留向上揭示 | 相机/机械臂高度横穿 |
| pedestal_down | 降低 0.065 m | 揭示主体下部，保留向下揭示 | 略低于初始相机的区域横穿 |

机器人使用有折叠余量的初态；左右横移分别预置对应底盘朝向。这些属于运行时配置，不属于任务的语义容差。`scenarios.py` 可重新生成目录；已有运行会保存独立任务与环境快照。

镜头开始前有 0.8 秒实际传感器预热，障碍物按首关键帧速度向过去线性延伸并持续运动，避免在 t=0 突然从静止跳到匀速。预热只提供已观测的历史，不向规划器暴露未来轨迹；它不计入镜头预算。困难预测最长前看 3.5 秒。

## 运行

从仓库根目录执行。系统 Python 用于 ROS/CPU，Isaac 使用 `env_isaaclab`，SAM 使用已有 `sam2` 环境。没有修改这三个环境的依赖。

```bash
# 单元与约束回归
PYTHONPATH=ros2_ws/jev_agent_wbc /usr/bin/python3 -m pytest -q ros2_ws/jev_agent_wbc/test

# CPU 集成对照；它不能替代 PhysX 实测
PYTHONPATH=ros2_ws/jev_agent_wbc OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /usr/bin/python3 -m jev_agent_wbc.runner --case all --mode both --pace .2 \
  --output /tmp/jev_cpu_run

# 真实 PhysX + 稀疏双 LiDAR + 参考/执行视频，一次启动完成全部场景
source /opt/ros/humble/setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
  /usr/bin/python3 ros2_ws/jev_agent_wbc/scripts/jvs_isaac_bootstrap.py \
  --isaac-python /home/w84384651/miniconda3/envs/env_isaaclab/bin/python \
  --runner ros2_ws/jev_agent_wbc/scripts/run_isaac.py -- \
  --case all --mode both --physics-hz 1000 --pace .2 --record \
  --output /tmp/jev_isaac_run

# Isaac 退出后，离线运行官方 SAM 2.1 large 视频预测器
/home/w84384651/miniconda3/envs/sam2/bin/python \
  ros2_ws/jev_agent_wbc/scripts/evaluate_sam2.py --runs /tmp/jev_isaac_run \
  --sam-root /home/w84384651/Documents/.RoboCine-Jev/sam21

# 汇总表、轨迹图、四项指标及权限实际使用图（绘图需 matplotlib）
/usr/bin/python3 ros2_ws/jev_agent_wbc/scripts/summarize.py /tmp/jev_isaac_run
/usr/bin/python3 ros2_ws/jev_agent_wbc/scripts/plot_results.py /tmp/jev_isaac_run
```

输出目录应为新目录，防止混入旧视频。`--case` 可单独选 orbit、movein、truck_left、truck_right、pedestal_up 或 pedestal_down；Isaac 还接受逗号分隔列表。`--mode strict` 固定原始目标，`--mode jev` 启用语义权限。两种模式使用相同底层动态安全控制，strict 也可能为硬安全偏离原始目标，并被同样评分。

验收时读取各组 `summary.json` 的 `completed`、`safety_pass` 和 `within_budget_success`，分别判断预算执行、安全及固定阶段完成情况；Kit 进程退出码不能代替这些指标。

默认使用有界 task-space 搜索与执行 WBC。`--allocation-preview` 可启用复制来的 shot-specific 底盘分配 MPC 进行消融；它不是 Jev 开关，最终选择应以记录的配置为准。`--pace` 是仿真秒/墙钟秒的上限，`.2` 给短镜头留出真实 API 响应时间；它不改变参考轨迹的 8 秒预算。运行同时报告实际墙钟时间和控制回调延迟，不能把锁步仿真结果宣称为实时控制保证。

Isaac 采用 headless、单 GPU、384×288、8 fps 双路测量视频、无 ROS 图、无 RTX LiDAR；障碍感知使用 PhysX 最近命中射线，机器人位置/速度和接触来自真实 PhysX。电机目标在物理子步内平滑变化，制动预测使用相同积分模型。等效转子惯量与驱动增益属于可记录的仿真执行配置，不改变固定评价的物理限制。降低物理频率前应复核加速度审计。原始物理限制仍为线速度 0.5 m/s、角速度 1.2 rad/s、线加速度 0.5 m/s²、角加速度 1.5 rad/s²、轮速 15 rad/s、轮加速度 20 rad/s²；逐轴机械臂限制见 `whole_body_qp.py` 与 URDF。

Isaac 5.1 内存 URDF 导入的 `/colliders`、`/visuals` 模板库会被标记为 USD class，防止它们在世界原点成为额外碰撞体和图像内容；机器人上的引用保持有效。每次运行断言修改前后真实机器人碰撞体列表完全一致，并保存 `import_collision_audit.json` 与初始射线身份审计。该处理遵循 [OpenUSD class 语义](https://openusd.org/release/glossary.html#class)，没有关闭机器人碰撞。

默认物理频率 1000 Hz，WBC 50 Hz、LiDAR 10 Hz。Isaac 规划使用 95% 速度、40% 加速度/轮加速度和 15% 底盘角加速度作为执行余量；固定物理评分仍使用上面的原始限制。`--acceleration-reserve` 与 `--yaw-acceleration-reserve` 属于运行时控制参数。参考进度、录像和评估不超过原始预算；不额外要求所有冗余关节在终点同时硬停。终点剩余速度与命令单独记录，ROS 适配器随后继续受约束制动，Isaac 镜头记录不包含后续停稳过程。

## ROS 2 接入

```bash
source /opt/ros/humble/setup.bash
colcon --log-base ros2_ws/jev_agent_wbc/.colcon_log build \
  --base-paths ros2_ws/jev_agent_wbc --packages-select jev_agent_wbc \
  --build-base ros2_ws/jev_agent_wbc/.colcon_build \
  --install-base ros2_ws/jev_agent_wbc/.colcon_install --symlink-install
source ros2_ws/jev_agent_wbc/.colcon_install/setup.bash
ros2 launch jev_agent_wbc cinematic_wbc.launch.py case:=movein mode:=jev \
  output_directory:=/tmp/jev_launch_test physics_hz:=1000
```

已有机器人/模拟器可运行 `cinematic_ros.launch.py task_file:=<绝对路径>`。所需运行时输入：`/joint_states`、`/base_controller/odom`、时间同步的 `odom -> base_link / LiDAR` TF、`/lidar/points` 与 `/lidar/front/points`，以及 `/cinematic/subject_pose`。输出只有 `/isaac_joint_commands` 一个 JointState 发布器，另有 `/cinematic/diagnostics`。主体 PoseStamped 应已变换到参考世界/odom 系；默认 launch 使用仿真时钟。

`scripts/smoke_ros.py --output <目录>` 提供私有 DDS domain 的合成传感器闭环测试。它只检验 ROS 传输、TF、启动和控制，不作为 Isaac 物理验收。控制间隔异常、过期传感器和不可认证命令会被记录；实际接入仍依赖执行端已有的独立失联制动机制。

## 固定评估与文件

每次运行保存原始任务、环境快照、评价协议哈希、源码/配置指纹。Jev 无法修改评价模块：

- Composition Error：SAM mask 质心归一化到图像宽高后的二维距离。
- Subject Scale Error：`abs(actual_area_fraction-reference_area_fraction)/reference_area_fraction`。
- Observation Angle Error：在主体坐标系中比较“主体指向参考相机”与“主体指向实测相机”的向量夹角。主体信息来自时间戳 LiDAR 观测；无有效位置/朝向时记为无效。
- Visual Smoothness：50 Hz 原始相机实测位姿的平移 jerk RMS 与世界系角加速度 RMS，另报避让控制切换、权限变更和阶段边界附近 ±0.3 秒窗口。没有为评分平滑信号。

参考与执行按同一原始时刻配对，不做动态时间规整。参考画面为原始相机路径的无遮挡场景，执行画面含机器人及移动障碍；初帧投影框只用于离线标识 SAM 主体，后续逐帧 mask 均由官方视频预测器产生。SAM 不在 Jev 决策数据中。

无 mask、空 mask、分割失败、画面边缘、遮挡、出画及未知可见性分别统计，不把失败算成零误差。独立报告接触、几何净距、逐物理子步的关节/轮速/加速度/侧滑、规划可行率、最小关节余量、阶段完成度、预算内成功率和控制耗时。阶段终点 0.05 m / 0.10 rad 是运行前固定的几何完成判据，不是 Jev 的容差；连续指标和全部失败仍保留。

每个 case/mode 目录包含：

- `summary.json`、`telemetry.json`、`physics.csv.gz`、`contacts.json`；
- `events.jsonl`：困难报告、完整权限、候选预测、实际采用参数、恢复及拒绝原因；
- `jev_api.jsonl`：实际请求/响应及延迟；strict 模式没有模型调用；
- `video/reference.mp4`、`video/execution.mp4`、对应 JPEG、配对时刻与 SAM 初始框；
- 离线后：`visual_metrics.json`、逐帧 mask NPZ 和少量人工复核覆盖图。

## 适用范围

安全证书基于机器人碰撞几何、当前观测及局部运动预测；它不等价于未观测环境的全局安全证明，也未提供覆盖全部关节空间的独立自碰撞证明。已跟踪物体使用观测尺寸构造保守占用体，防止近处缺失表面被误认为空闲。CBF 半空间无解时，可尝试保持全部执行器限制、重新通过连续动态距离证书的紧急控制；无法认证则报告失败，不把任意零命令标为安全。

task-space 规划目前搜索六维偏移、C2 激活/恢复的有限候选族，不是任意拓扑场景的完备全局规划器。完整参考目标误差与实际 WBC 跟踪残差均保留；目标在 Jev 权限内不代表物理执行没有跟踪误差。目标方向保护采用 20 ms 网格检查，连续碰撞证书则使用独立的速度界限。

当前日志中的规划可行率表示预测控制通过数值安全及执行器约束；候选仍可能有较大 task-space 跟踪残差，它不能等同于目标轨迹已可精确实现。需要进一步约束预测实际轨迹与 Jev 权限的一致性，并改善物理执行闭环和可见性处理。

当前运动物体跟踪器估计水平平移，垂直速度固定为零；上下升降运镜使用位于机械臂高度的横穿障碍。当前主体静止且绕竖轴对称，主体朝向采用固定标定；旋转主体和通用主体自动关联尚未做验证。Isaac 中主体点云通过物体标识辅助关联，三维位置仍由射线命中点估计；缺少当前有效主体点云时角度指标记为无效。射线适配器利用已知机器人刚体身份过滤自身回波，并保留首命中的遮挡关系；ROS 点云接入使用几何自身过滤。两者均未使用外部障碍物身份或未来轨迹来规划。LiDAR 是理想稀疏射线，未模拟 RTX 材质、运动畸变或真实噪声。出画判断使用主体中心，边缘 mask 另行统计；遮挡使用有限主体表面射线。几何净距按控制步记录，接触与执行器审计覆盖每个物理子步。四项几何指标只能作为意图保留的代理，不能直接充当完整语义意图评分。

复制来源及原始文件哈希见 `VENDORED.json`。本包中的修改只作用于新目录，已有 `jvs_mock` 实现保持独立。
