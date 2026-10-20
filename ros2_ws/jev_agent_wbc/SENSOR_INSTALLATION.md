# 三款机器人 Camera 与 LiDAR 初始安装

三个独立机器人入口均安装 **1 个运镜 Camera**。根据实际遮挡测试，当前默认候选的
LiDAR 数量为 **G1 4 个、Go2 3 个、Ridgeback 3 个**，分别包含一个主扫描器和辅助扫描器。
当前默认候选仍未通过全姿态相机视野／安装干涉与闭环避障验收，不能据此保证硬件安全。
配置为 [robot_sensor_mounts.json](config/robot_sensor_mounts.json)，实现为
[robot_sensor_rig.py](jev_agent_wbc/robot_sensor_rig.py)。安装参数属于仿真原型，
未绑定真实传感器型号，支架质量、惯量、碰撞外形及线缆尚未建模。

| 平台 | Camera 父部件与局部位置（m） | 镜头前方／画面上方 | 主 LiDAR 父部件与局部位置（m） |
|---|---|---|---|
| G1／AGILE | `right_wrist_yaw_link`，`[0.20, 0.04, 0.11]` | 腕系 `+X`／`+Z` | `torso_link`，`[0.10, 0, 0.32]` |
| Go2＋Piper | `end_effector`，`[0.075, -0.04, 0.12]` | 末端系 `+X`／`+Z` | `base`，`[-0.25, 0, 0.36]` |
| Ridgeback＋UR5 | `ur_arm_wrist_3_link`，`[0, 0, 0.08]` | 法兰系 `+Z`／`+Y` | `base_link`，`[-0.22, 0, 0.43]` |

位置均相对于真实父刚体。Go2 当前主平台为已有匹配 WBC 策略的 Piper，D1 不复用此标定。
Go2 主 LiDAR 高于机械臂基座、略偏后；Ridgeback 主 LiDAR 在机械臂基座后方。
G1 的头、肩、手臂仍可能形成遮挡；上述位置均需经姿态检查调整。

光学坐标为 **+X 向右、+Y 向下、+Z 拍摄前方**。`T_parent_optical` 明确完整安装变换；
其子 USD Camera 绕 X 轴 180°，将 USD 的 `-Z` 拍摄方向对齐到光学 `+Z`。
UR5 与另外两款机器人的末端轴向不同。默认分辨率 640×480、水平 FOV 75°，可修改配置。

主 LiDAR 是 PhysX 最近命中射线查询：水平 360°、180 个方位、71 个仰角（-80° 至 +60°，
包含水平面），每个扫描器共 12,780 束，默认量程 0.05–10 m。瞬时扫描尚未模拟 RTX、真实扫描时序、
强度、噪声和材质反射。首先命中本体时保留遮挡，不继续透过机器人查询；
最小量程内本体命中也保留。未命中、近距离盲区、本体遮挡不输出为环境障碍点。
原资产自带的传感器描述保留。最初一相机、一主查询扫描器的配置归档于
[robot_sensor_mounts_initial.json](config/robot_sensor_mounts_initial.json)，可用 `--sensor-config` 对照。

## 在现有物理运行中检查

三个 `run.sh` 默认创建传感器 USD 安装描述，写出 `sensor_mounts.json`。
`--sensor-audit` 在运动中采集扫描，`--sensor-rgb` 在运动结束后检查相机图像：

```bash
# 仓库根目录执行；每次使用不存在的输出目录。
ros2_ws/jev_agent_wbc/Robot_G1/run.sh \
  --seconds 4 --sensor-audit --sensor-rgb --output results/sensor_smoke_01
ros2_ws/jev_agent_wbc/Robot_Go2/run.sh \
  --seconds 4 --cases stand,coordinated --sensor-audit --sensor-rgb \
  --output results/sensor_smoke_01
ros2_ws/jev_agent_wbc/Robot_Ridgeback_UR5/run.sh \
  --probe --sensor-audit --sensor-rgb --output <新的输出目录>
```

G1 包装器切换到机器人目录，相对输出路径也相对于该目录。
Go2、Ridgeback 相对输出路径相对于当前工作目录。可全部使用绝对路径。
短测试仅作诊断；Go2 4 秒前缀也不足以证明协调动作充分展开。原策略、驱动、步长和 `pace=1` 保留。

扫描按 2 Hz 记录，开销计入原运行耗时。观测采用 PhysX／Lab 实测刚体位置和完整四元数旋转，
包含四足俯仰、侧倾，不只采用 yaw，也不依赖可能滞后的 Fabric USD 位姿。
多传感器在同一物理时刻顺序查询，期间不推进或修改场景；每份扫描同时提供局部点、
世界点、`T_world_lidar` 和采集时间。Go2 独立重置场景使用累计仿真时钟和 `pose_label`。

RGB 检查在运动与计时结束后进行：先保存实际视图，再在光学 `+Z`／`-Z` 放置红／绿无碰撞探针，
检查中心像素能否看到前方红色探针。检查不推进物理，结束后删除探针。
这仅验证当前渲染方向，不能证明整段构图、无自遮挡或 Agent 输入已接通。
渲染异常独立记录在 `camera_render_error.txt`，并令 `camera_render_verified=false`；
原物理验收退出码与传感器诊断分别判断，物理通过不能替代传感器验证。

| 产物 | 含义 |
|---|---|
| `sensor_mounts.json`、`sensor_rig_source.py` | 完整安装参数、变换、配置／源码哈希、运行源码副本 |
| `sensor_samples.jsonl` | 采集时间、Camera 光学位姿、LiDAR 世界变换／点云、逐束命中与本体遮挡掩码 |
| `sensor_audit.json` | 采样数量、持续遮挡水平／全仰角方位、渲染结果和未验证范围 |
| `camera_rgb.png` | 最终物理场景中任务相机实际 RGB 诊断图 |
| `camera_direction_probe.png`、`camera_render_check.json` | 前后探针图、中心像素、朝向检查结果 |

RGB 是单帧诊断，整机闭环交付视频仍遵循项目原有 Representation 流程。
避障 Agent 尚未连接三个新平台的传感器。

## 不启动 Kit 的安装检查

已有 OpenUSD Python 环境可从 `ros2_ws/jev_agent_wbc` 执行：

```bash
python scripts/author_robot_sensors.py --robot all --output <新的输出目录>
```

脚本引用本地原始资产，输出各款 `*_sensors.usda` 与安装清单，不改写原模型，
不执行物理或渲染。普通 Python 缺少 `pxr` 时需使用提供 OpenUSD 的环境。

## 姿态覆盖与辅助 LiDAR

报告始终保留 `full_motion_range_verified=false`。G1 行走策略保持默认手臂姿态，
Go2 协调目标只覆盖有限范围，原运动验收不足以验证全机械臂构型。
完整检查需覆盖伸展、左右旋转、折叠、极限腕姿、底盘转向和四足俯仰／侧倾，
同时检查实际图像与逐束遮挡。水平 360° 表示发射方位范围，不等于无遮挡。

根据逐束掩码确定持续盲区后，可在完整配置的 `lidars` 数组增加腰部、两侧或低位传感器，
各自设置唯一 `name`、真实刚体 `parent`、局部平移与 `rpy_deg`。
启动时用 `--sensor-config <完整JSON路径>` 应用；多扫描器仍输出同一世界坐标。
数量应由实际视场、近距离盲区、低矮障碍可观测性及全姿态测量决定。

2026-10-19 首轮完成三款实际 USD 组装与父部件／光轴／完整 TF 检查。
随后确认 GPU 设备只在执行沙箱外可见，已经在 RTX 5090／Isaac Sim 5.1 中执行实际渲染、
移动扫描及有限姿态覆盖测试。不能将沙箱内 `nvidia-smi` 失败解释为宿主机没有 GPU。

## 实际渲染与覆盖测试（2026-10-19 至 20）

新增 [validate_robot_sensor_coverage.py](scripts/validate_robot_sensor_coverage.py) 建立隔离诊断场景：
G1 采样腰部、双臂全部 17 个关节，Go2 与 UR5 各采样 6 个臂关节；每个关节包含两端点
（离硬限位 0.02 rad），再加入固定种子的 64 个 Sobol 组合姿态及默认姿态。
共 G1 99 个、Go2 77 个、Ridgeback 77 个有限姿态。G1／Go2 还加入约 ±0.2 rad 的
机身 roll／pitch 和全周 yaw；这些不是四足所有腿部／接触状态，也不是连续关节空间证明。

每个姿态在半径 0.7 m、1.5 m，目标中心高度 0.06、0.20、0.50、0.90、1.35 m
布置 36 个 12 cm 立方障碍，分别扫描。只将最近命中且不小于最小量程的目标返回计入检测；
本体遮挡不能穿透，最小量程内命中不能作为有效环境检测。
每个姿态保存实际 RGB、语义分割本体掩码、64×48×44 mm 相机外壳代理干涉、
半边长 5 mm 的 LiDAR 原点检查盒与机器人的重叠、逐束命中点／碰撞身份／采集时间／世界 TF。
碰撞检查包含 USD 实例代理网格；不只检查非实例化网格。

扫描可复用项目已验证的原生最近命中查询；启动时逐束对照公开标量 API。
该后端最大查询距离为 4.5 m，本测试仅评价上述 0.7／1.5 m 环，不据此证明配置标称 10 m 量程。
所有查询同步完成后才改变场景。另用 [analyze_robot_sensor_coverage.py](scripts/analyze_robot_sensor_coverage.py)
从 NPZ 命中身份重算检测数量、核对目标几何／采集 TF／RGB 哈希／遮挡掩码。
原始本体返回保留作诊断，融合环境点时排除本体；PhysX 圆柱体自命中坐标的亚毫米数值偏差独立报告。

实际测试发现原单主 LiDAR 的低位／侧向盲区和 G1 手指遮挡；尝试过的安装清单和失败数据均保留。
[robot_sensor_mounts_coverage.json](config/robot_sensor_mounts_coverage.json) 是经过本轮测试的扩展候选：
G1 4 个、Go2 3 个、Ridgeback 3 个扫描器，垂直采样 -80° 至 +60°、间隔 2°。
相机偏移也经过有限网格搜索调整。这些数量服务于本轮具体测试目标，未证明是最少硬件数量，
尚未指定能够实现该视场的真实 LiDAR 型号。当前默认已合并这组实测候选；最初一主扫描器配置归档留作对照。
安装数量的增加不能替代当前仍未通过的相机与闭环避障验收。
Ridgeback 早期试过 5 个低位／角部发射点，但部分近场目标会直接包住发射原点，
最小量程检查正确地拒绝这些返回；之后改测顶部对角位置。旧数据未删除，不能混成同一配置的成功率。

```bash
# 在宿主 GPU 执行环境，从本包目录运行；输出目录必须不存在。
OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/w84384651/miniconda3/envs/env_isaaclab/bin/python \
  scripts/validate_robot_sensor_coverage.py --robot go2 --poses 64 \
  --config config/robot_sensor_mounts_coverage.json --output <新的输出目录>
```

姿态检查固定基座、关闭重力并直接放置关节状态，以隔离几何问题；不称为原控制器的动态执行。
原控制器另在 `motion_g1`、`motion_go2`、`motion_ridgeback` 中执行移动扫描与最终 RGB 检查，
保留原物理步长、驱动、控制器和 `pace=1`。诊断计算开销照实计入运行墙钟。

所有实验位于 [sensor_coverage_20261019](results/sensor_coverage_20261019/)。
覆盖测试的脚本退出 0 仅表示记录完成；是否漏检、相机是否无遮挡、外壳是否干涉必须分别读取 summary／审计结果。
相机固定在末端并保留原关节上下限，仍可能在合法关节姿态下朝向机身或接近其他部件。
不能通过缩小关节范围或隐藏这些失败帧宣称“全姿态无遮挡”。
支架、实际传感器外形／质量、线缆扫掠和闭环避障 Agent 仍需独立验证；目前不作这些保证。
