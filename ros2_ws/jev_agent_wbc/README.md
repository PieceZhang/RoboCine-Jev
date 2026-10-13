# Jev cinematic WBC

独立 ROS 2 包，用于差动底盘与六轴机械臂的运镜修复。默认 Agent 已改为 **选择修复族 → 有界整机预测 → 审批完整候选 → 最新状态复核**。WBC 持续执行并检查安全；它不等待 Jev、SAM2.1 或数学进程。原始相机轨迹、修改目标和实际执行始终分开。

论文定位、最小实验矩阵与统计方案见 [PAPER_DESIGN.md](PAPER_DESIGN.md)。最新的约 1.5 小时优化记录、demo 和卡点见 [优化报告](results/optimization_20261013_0835/REPORT.md)。前一轮重构验证见 [VALIDATION.md](VALIDATION.md)；旧版历史实验仍保留在 [EVALUATION.md](EVALUATION.md)，不可当成新架构的结果。当前系统尚未达到全部动态场景的安全完成标准，也未证明 Jev 优于规则基线。

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
4. Jev 用一次带拒绝选项的 Choice 审批完整方案。输入包含预测**实际执行**的构图、距离尺度代理、观察角、阶段进度、最差时刻和后续余量；不发送点云或图像，不做六轮数值权限搜索。
5. 批准绑定候选哈希、原参考和父方案。数学进程从新快照重预测；控制侧只做有界的状态、占用、尺寸/速度变化与证据门禁。迟到方案不能偷偷改变生效时刻。
6. 执行反馈可触发新的取舍。底层仍独立通过约束与动态扫掠证书，无法认证时报告失败。镜头时间不会暂停或延长。

当前 API 默认为 `http://127.0.0.1:8000/v1/systemone`、`model="typesafe-jev"`（可通过 `--model` 显式切换），采用本机服务的 Choice 协议。候选使用短别名，精确轨迹及别名到哈希的绑定保留在日志中。模型概率只记录为决策信息，不作碰撞或传感器可信度证明。

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
