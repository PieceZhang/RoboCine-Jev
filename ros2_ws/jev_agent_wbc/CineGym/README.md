# CineGym

用于整机避障 Agent 评测的随机运镜环境。代码与 `scenarios/` 同级，沿用现有
`task.json`、`runtime_scene.json`、`OriginalReference`、全 8 部件碰撞几何与 LiDAR 观测链路。

提供两层接口：

- **现有 Agent 的 CPU / PhysX 闭环**：`run_closed_loop`、`start_closed_loop` 或命令行 `run`。
  两种后端读取完全相同的已保存 episode；`--mode jev --api-url ... --model ...` 接入现有 Jev 服务。
- **自定义低层策略的 CPU / PhysX 交互**：`CineGymEnv(backend="cpu"或"physx").reset/step`，
  输入真实单位的底盘及关节速度，返回实测状态、当前参考和传感器观测。
  这是 Gym 风格接口，不依赖 Gymnasium，也不是可直接传给 Stable-Baselines 的 Gymnasium Env。
  PhysX 的 `step` 通过独立进程提交动作并等待下一份观测；物理世界持续前进，不等待调用者。

## 随机化与覆盖

14 类镜头包括原有 6 个基础运镜、原有 6 个组合的动作序列、`moveout` 和随机 2–4 动作组合。
每次随机化初始朝向与位置、机械臂姿态、主体距离、动作幅度、双向环绕、时长与速度分布。
上升/下降仍伴随前移，使差速底盘能够参与；组合通过共享端点衔接，没有待补全的转场空段。
参考位置使用原有五次 smoothstep 采样，执行时仍由 `OriginalReference` 插值。

每个场景至少含低、中、高三个障碍物，分别针对底盘、中段机械臂、腕部/相机：

| 档位 | 运动球体默认数量 | 目标干涉深度 | 抽样速度尺度 |
| --- | --- | --- | --- |
| easy | 3 | 0.015–0.035 m | 0.16–0.30 m/s |
| medium | 3–5 | 0.035–0.065 m | 0.22–0.42 m/s |
| hard | 5–7 | 0.055–0.090 m | 0.30–0.58 m/s |

默认 `generation_style="natural"`（v3）：easy / medium / hard 分别安排 1 / 2 / 3 个实际
冲突物体，其余物体从附近通过。三个高度都有覆盖，但不要求每个物体都撞向机器人。
全名义轨迹的采样侵入深度必须满足档位上限；路过物体的名义净距至少为 0.12 m。
`generation_style="legacy"` / `--generation-style legacy` 保留旧的全物体干涉生成方式，
旧方式只有显式设置 `enforce_global_penetration=True` 才限制全程采样侵入深度。

natural 的轨迹是独立世界坐标运动，使用 C2 样条、局部平滑转向/停走/折返，首末不强制
与镜头同步静止。用 Bernstein 控制点凸包约束连续时间速度、加速度和分段 jerk 上界：

| 档位 | 速度上界 | 加速度上界 | jerk 上界 | 首次名义接触最早时间 |
| --- | --- | --- | --- | --- |
| easy | 0.42 m/s | 0.8 m/s² | 3 m/s³ | 3.0 s |
| medium | 0.588 m/s | 1.2 m/s² | 6 m/s³ | 2.5 s |
| hard | 0.812 m/s | 2.0 m/s² | 12 m/s³ | 1.2 s |

显式配置的速度/加速度上限如果更小，则采用更小值。默认每个动作 8–12 s；组合镜头的
冲突分布到各动作内部，避免全挤在动作交界处。记录实际首次接触、冲突持续时间、同时
干涉数量及重采样原因。上述接触和净距筛选仍是 20 ms 采样，不能冒充连续安全证明。
6 种运动为 `crossing` 横穿、`curved` 弯曲穿越、`stop_go` 停留后继续、
`reverse` 折返、`wave` 波形侧移、`diagonal3d` 小幅升降穿越。球形代理与原 demo 一致，
半径独立随机，默认 0.075–0.125 m；高度随目标部件及姿态变化。
高、中障碍代表外部运动物体/部件的球形碰撞代理，不模拟其支架。

生成器先做无障碍名义 WBC rollout，再在**同一时间的整机几何**附近拟合独立路线：

- 名义运镜跟踪误差筛选：采样最大位置误差 ≤ 0.045 m、姿态误差 ≤ 0.08 rad。
  natural 还必须通过原完整运镜验收、自碰撞检查和终态保持；不能仅靠位置跟踪误差入选。
- 冲突物体必须对指定高度部件形成负表面净距；保存见证时间和部件，路过物体单独标记。
- 初始及终态名义整机净距 ≥ 0.12 m，障碍之间净距 ≥ 0.025 m，并避开地面及主体。
- 中/高难度增加障碍数量与事件重叠机会，实际最大同时干涉数量写入 metadata。
- 默认按 0.02 s 审计。以上是离线采样筛选，**不等于连续安全证明，也不保证一定存在可行避障解**。
  难度为生成分布标签，不是经 Agent 成功率标定后的难度。

对仅球体场景，可选 `GymConfig(require_avoidance_witness=True)` / `--require-avoidance-witness`
要求额外的离线可行解；默认混合场景强制要求此检查。在动作内部调整时钟
（双动作组合独立调整，更长组合使用共同的时钟参数），保持动作顺序、边界和截止时间；按 CPU 的
20 ms 恒速度命令模型重新积分，检查速度、命令变化率、轮速、关节限位、自碰撞、原画面
验收与障碍遮挡。表面净距用采样值减去运动速度 Lipschitz 上界证明，保持原 0.025 m 外部
及 0.004 m 自碰撞门槛。见证及其真值信息只进入离线 metadata，不提供给 Agent。
这是 CPU 模型下的可解性证明，不保证传感器噪声、遮挡下的在线可观测性，也不代替 PhysX
或硬件验证；有限时钟族找不到解只表示未验证，不表示场景无解。此选项会改变场景分布，
必须与普通 natural 套件分别报告，并对优化前后使用完全相同的 episode 哈希。

障碍轨迹生成后固定，不追踪实际机器人。Agent 只得到当前/过去的 LiDAR 返回与跟踪结果；
未来障碍关键帧和真值几何仅用于场景执行、离线设计和评测。

v3 在上述运动球体之外默认加入 **2 个落地长方体**（hard 为 3 个）。宽 0.22–0.62 m、
深 0.20–0.55 m；低箱高 0.16–0.38 m，高箱高 0.55–1.05 m，朝向独立随机，底面严格为 z=0。
箱体是静态环境边界，放在运镜路线附近，名义整机净距至少 0.12 m，用于限制动态避障时
可用的侧向空间。它们不与运动球体或主体相交；运动球体仍负责触发实际避障事件。

混合场景**强制通过完整可行解检查**：全部球体和全部箱体必须保留至少一个共同的
CPU 整机时序见证，并通过原运镜验收。这可排除在该 CPU 模型下堵死所有可行路线的场景，
不等于在线 Agent 必然能从有限观测找到该解。失败则重采样，不输出未验证的混合场景。
`ground_boxes=False` / `--no-ground-boxes` 可显式恢复仅球体分布；legacy 保持原有球体场景。

CPU LiDAR 对真实朝向的箱体表面做精确射线求交；只将命中点交给感知模块。
CPU 真值碰撞和离线见证使用覆盖箱体完整体积的球并集，每个子格边长至多 0.16 m，
子格外接球保证不会漏掉箱体角落；因此箱体净距是保守下界，代理相交不能直接称为真实
箱体接触。场景 JSON 保存 `shape="box"`、`size=[宽,深,高]` 和 `yaw_rad`；`radius` 仅为
兼容字段，混合场景审计不把它当作单球替代箱体。PhysX 创建实际立方体碰撞体；本轮新增
箱体已进入后续官方 Jev 的 PhysX 闭环对照；实际成功率与失败范围见
`../results/cinegym_physx_iteration_20261020_0211/report.txt`，不能把几何可行见证当作在线成功。

## 环境与生成

在仓库根目录运行；依赖与已有 WBC 相同：NumPy、SciPy、OSQP。
本机 `/usr/bin/python3` 已包含这些依赖，基础 Conda Python 不一定包含。

```bash
export PYTHONPATH="$PWD/ros2_ws/jev_agent_wbc${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# 默认均衡遍历 14 类镜头 × 3 档难度；输出目录必须不存在。
/usr/bin/python3 -m CineGym generate \
  --output /tmp/cinegym_benchmark --count 42 --seed 100 --split test

# 固定难度、随机镜头；--duration 为每个主动作的时长范围。
/usr/bin/python3 -m CineGym generate \
  --output /tmp/cinegym_random --count 100 --seed 1000 \
  --random --difficulty hard --duration 6 10 --obstacles 6
```

默认均衡模式覆盖全部难度；`--difficulty` 在 `--random` 下生效。
可用 `--family` 固定某类镜头。`train/validation/test` 使用不同随机流；相同 seed、split、
配置和代码/数值依赖得到相同场景。每个 episode 有独立 seed，因此不依赖批量生成顺序。
输出包含 `catalog.json` 及每个场景的三个文件：

- `task.json`：严格保持上游四字段契约。
- `runtime_scene.json`：初始状态、主体、各障碍物的 Hermite 位置/速度关键帧。
- `metadata.json`：种子、配置、参数、重采样原因、几何审计、输入和源码 SHA-256。

只测试单运镜及两个主动作组合时，使用受限平衡套件入口：

```bash
/usr/bin/python3 ros2_ws/jev_agent_wbc/scripts/benchmark_cinegym_cpu.py generate \
  --output /tmp/cinegym_simple_medium --count 40 --seed 9700 --split test \
  --generation-style natural

# 可选：生成仅球体、仍要求独立 CPU 可行解证据的另一套场景。
/usr/bin/python3 ros2_ws/jev_agent_wbc/scripts/benchmark_cinegym_cpu.py generate \
  --output /tmp/cinegym_solvable --count 20 --seed 19700 --split test \
  --no-ground-boxes --require-avoidance-witness --max-attempts 48

/usr/bin/python3 ros2_ws/jev_agent_wbc/scripts/benchmark_cinegym_cpu.py run \
  --suite /tmp/cinegym_simple_medium --output /tmp/cinegym_simple_medium_jev \
  --mode jev --lead 3.2 --budget 2.5
```

该入口仅采样 easy/medium，组合固定为两个动作。生成前保存独立的 `*_source`，
每组执行前保存 `frozen_code`，避免迭代期间源码变化混入同一套场景或运行。
通用接口的 `GymConfig(combo_actions=2)` / `--combo-actions 2` 只限制 `random_combo`，
不会改变已有三/四动作固定模板的定义。

到达重试上限会明确报错，不会用未通过筛选的样本补齐数量。批量生成中断时 catalog 标为
`complete=false`；可显式选择已生成案例执行，不能把不完整套件作为完整基准。

## CPU 闭环

```bash
# 无网络基线，调用原有 TPR-WBC；碰撞/约束失败按实际结果保留。
/usr/bin/python3 -m CineGym run --suite /tmp/cinegym_benchmark \
  --case all --backend cpu --mode strict --output /tmp/cinegym_cpu

# 现有 Jev Agent。API key 使用已有服务的环境变量/文件机制，不写入场景。
/usr/bin/python3 -m CineGym run --suite /tmp/cinegym_benchmark \
  --case all --backend cpu --mode jev --output /tmp/cinegym_jev \
  --api-url https://api.typesafe.ai/v1/systemone --model jev-1.14.0 \
  -- --single-candidate --candidate-proposals 1 --candidate-budget 1
```

`--case` 支持 catalog 中的名称或逗号分隔名单。`--` 后转发已有后端参数，例如
`--preplan`、`--candidate-wall-budget`，但不允许覆盖场景身份、输出、pace 与物理默认值。
每个案例输出独立 `backend.log`、`cinegym_run.json` 以及原后端的 telemetry、events、summary。
`--timeout` 为每案例进程墙钟上限，默认 600 s；超时不会当成完成。
Jev 默认使用官方 TypeSafe 地址、`jev-1.14.0` 与仓库根目录 `typesafe_Jev_api_key` 文件；
只有显式传入 `--api-url` 才更换服务。密钥内容不写入日志或场景。

## PhysX 闭环

需要安装 Isaac Sim、可用 NVIDIA GPU，以及本项目原有 ROS/xacro 环境。启动器自动为 xacro
补齐 `/opt/ros/humble` 下的 Python 路径与可执行路径，保留用户原有路径；Isaac 仍经原
bootstrap 使用自带 ROS ABI。自定义 ROS 安装可设置 `CINEGYM_ROS_PREFIX`。

```bash
export PYTHONPATH="$PWD/ros2_ws/jev_agent_wbc${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CINEGYM_ISAAC_PYTHON=/home/w84384651/miniconda3/envs/env_isaaclab/bin/python
# 此兼容目录存在时启动器会自动选用；也可显式指定其他经过验证的数值环境。
export JEV_PHYSX_NUMERIC_SITE="$PWD/ros2_ws/jev_agent_wbc/results/physx_cpu_parity_20261016/physx_numeric"

/usr/bin/python3 -m CineGym run --suite /tmp/cinegym_benchmark \
  --case all --backend physx --mode strict --output /tmp/cinegym_physx
```

也可用 `--isaac-python /path/to/python` 指定解释器。`--dry-run` 检查输入并打印实际 argv，
不会启动模拟器。PhysX 保持 **100 Hz 物理、pace=1、原始物理相机反馈**，复用已有有界异步
控制、命令超时及有界恢复机制；物理时间不会为了等 Agent 返回而暂停。
CineGym 默认从原始任务在线执行，与 CPU 的默认启动方式一致；需要开拍前规划时可显式传入
`-- --startup-preplan`。启动准备与镜头内重规划的时间应分别记录。
先用 `-- --duration-limit 0.4` 做短诊断，再跑完整镜头；诊断前缀不计为整镜头成功。
设置 pace=1 不代表实测达到实时，实际墙钟性能仍由后端记录。

2026-10-20 的实验配置额外启用以下选项（仍需单独验收运镜质量）：

```bash
/usr/bin/python3 -m CineGym run --suite /tmp/cinegym_benchmark \
  --case all --backend physx --mode jev --output /tmp/cinegym_physx_measured \
  -- --compiled-geometry --certificate-recovery --session-command-mode velocity
```

`--compiled-geometry` 使用 Isaac Python 中可选的 Numba CPU 几何核，保持原几何和净距门槛。
`--certificate-recovery` 对暂时无法证明安全的指令请求制动并有界等待；必须有新鲜传感绑定的
新指令、相关障碍的新观测及全部当前安全检查，才能恢复。丢失轨迹超时删除不构成清空区域的
证据；真实接触和恢复超时仍锁止。制动请求本身不等于制动路径已证明安全。

Agent 的阶段检查点仅适用于中间运镜已通过观测质量检查的组合任务；它不修改原运镜期限，
不清除物理故障，也不能覆盖最终原始相机全轨迹审计。实验中的候选 agent 改进与 PhysX
工程改进分别列在 `../results/cinegym_physx_iteration_20261020_0211/improvement_categories.json`。


如添加 `--record`，原始 RGB 是测量数据。交付影片应再使用现有
`scripts/render_agent_demos.py` 的 Representation 渲染流程，保留完整整机表示。

## Python 调用

```python
from CineGym import GymConfig, generate_episode, run_closed_loop, start_closed_loop

episode = generate_episode(31, GymConfig(family="random_combo", difficulty="hard"))
case = episode.save("/tmp/cinegym_case31")
cpu = run_closed_loop(case, "/tmp/cinegym_case31_cpu", backend="cpu", mode="strict")

# 非阻塞启动；poll 只读取状态，不影响物理执行。
with start_closed_loop(case, "/tmp/cinegym_case31_physx",
                       backend="physx", mode="jev",
                       isaac_python="/path/to/isaac/python",
                       api_url="https://api.typesafe.ai/v1/systemone") as session:
    status = session.poll()
    result = session.wait(timeout=600)
```

自定义策略使用相同的 `reset/step` 形式（零速策略仅演示接口，不能保证避障）：

```python
import numpy as np
from CineGym import CineGymEnv, GymConfig

with CineGymEnv(GymConfig(family="movein", difficulty="easy"), backend="cpu") as env:
    obs, reset_info = env.reset(seed=7)
    while True:
        action = np.zeros(8)  # 替换为 agent(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(info)
            break
```

换成 PhysX 时，循环无需改写，只替换环境构造：

```python
env = CineGymEnv(
    GymConfig(family="movein", difficulty="easy"), backend="physx",
    output="/tmp/cinegym_interactive",
    isaac_python="/home/w84384651/miniconda3/envs/env_isaaclab/bin/python",
)
```

也可向两个后端的 `reset(episode=Episode.load(...))` 传入同一场景，以确保输入相同。
每次 PhysX reset 都启动独立会话，结果保留在 `episode_0001/`、`episode_0002/` 等新目录。
`close()` 有界关闭该会话与其控制进程；不要在旧结果目录中重新创建新环境。

`obs` 包含时间、9 维实测状态、8 维上一控制、相机位姿、当前参考位姿/速度、已知任务与
LiDAR snapshot；不含障碍物关键帧。`action=[v, omega, dq1..dq6]` 的单位为 m/s、rad/s。
速度、加速度、轮速及关节限位越界会抛 `ValueError`，不悄悄裁剪策略输出。CPU 不推进该动作；
PhysX 不接受该动作，但后台物理时间仍继续。PhysX 额外返回实测完整底盘旋转、高度和速度。
`env.limits` 为当前后端的动作约束；PhysX 会在 reset 时同步实际配置（包括加速度预留），
同时在 `obs.action_limits` 中给出这些静态约束。
CPU 默认控制周期 0.02 s，周期内每 ≤0.005 s 检查整机净距；到达终点或碰撞后 terminated，
没有额外时间截断。PhysX 物理周期 0.01 s、控制周期 0.02 s，镜头后还执行原有 0.4 s 保持。
默认 reward 为 `-观测间隔*(位置误差 + 0.2*姿态误差) - 新碰撞指示`，用于诊断；
正式镜头验收仍使用原有闭环结果，不能把 reward 当作镜头成功。
`info` 中净距等为评测真值，训练/测试策略不应读取。CPU 没有物理接触响应。

PhysX 交互的时间与故障语义：

- `reset` 等待设备初始化和第一份实测观测；之后物理时钟独立前进。
- `step` 发送针对上次返回观测的动作。序号、仿真时间和墙钟时间保持原值，不能把迟到动作
  重新标成“当前动作”。返回的下一份观测可能尚未包含该动作的执行，不能假设一次 step 恰好推进一帧。
- `info.submitted_action_sequence` 与 `control_status.accepted_action_sequence` 可用于检查接受状态。
  队列只保留最新消息，慢策略可能跳过观测。缺省命令期限仍为 0.1 s。短时缺失进入有界制动等待，恢复必须同时满足新鲜观测、
  新命令和当前物理安全检查；迟到命令不能解除制动。硬安全故障仍锁存。
- 原有当前状态扫掠、速度变化认证、驱动限幅和物理接触审计全部保留。策略动作不会直接绕过安全层。
- 默认通信故障/超时制动返回 `truncated=True`；到达完整镜头终点或检测到碰撞返回 `terminated=True`。
  诊断可用 `terminate_on_fault=False`、`terminate_on_collision=False` 继续观察停车后的物理演化，
  这不会解除制动，也不会把故障变成成功。诊断前缀结束归为 truncation。
- 评测净距/接触反馈在下一份观测时作为 **info** 提供，`evaluation_delay_s` 标明延迟；
  未获得审计时为 `None`，不当作安全。终态完整结果以原后端 `summary.json` 为准。
- 不要在控制循环中插入等待、渲染或耗时计算；超过默认期限会如实触发故障。
  `pace=1` 是目标速度，不保证机器实际达到实时。

## 结果与回归测试

```bash
/usr/bin/python3 -m CineGym report --runs /tmp/cinegym_cpu --output /tmp/cinegym_report.json
PATH=/opt/ros/humble/bin:$PATH /usr/bin/python3 -m pytest -q \
  ros2_ws/jev_agent_wbc/CineGym/tests \
  ros2_ws/jev_agent_wbc/test/test_scenario_trajectories.py \
  ros2_ws/jev_agent_wbc/test/test_realtime.py \
  ros2_ws/jev_agent_wbc/test/test_physx_camera_views.py
```

报告按后端、运镜类别、难度分别汇总完成数、镜头成功率、安全运行完成率、全部已审计案例的
碰撞率，以及已完成案例的跟踪误差；后者不能代表失败前缀的误差分布。
保留原始净距、物理接触、控制延迟和实时指标；CPU 和 PhysX 不合并成一个成功率。
未执行、诊断前缀、缺少证据及仍有未响应官方 API 请求的试验记为 unverified，列出数量且不混入
成功/失败分母。报告也保留失败前缀的原始指标，避免只看成功案例。
进程墙钟超时即使保存了部分 summary，只要没有完整执行或确定的执行/安全失败证据，
也记为 unverified；明确失败或已完整结束的结果不会仅因后续关闭超时而被抹掉。

已有模块的兼容扩展：`RuntimeScene` 支持内存初始化与逐障碍物半径；`RealtimeEngine` 支持
可选外部控制 worker 和故障后的只读观测流；`run_isaac.py` 提供显式 `--cinegym-bridge` 接入。
不启用 bridge 时仍使用原有 Agent。原 12 个 demo 文件保持原样。

可用 `python -m CineGym.probe_interface --help` 查看接口诊断脚本。它使用微小底盘速度测试
动作与实测状态的闭环，`--stall-after 1` 会主动停顿 0.35 s 检查锁存制动；不是避障策略。
已完成的生成覆盖、CPU / GPU 实测及其限制见 [VALIDATION.md](VALIDATION.md)。
