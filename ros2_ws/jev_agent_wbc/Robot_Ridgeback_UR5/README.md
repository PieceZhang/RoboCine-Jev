# Ridgeback＋UR5 本地 Isaac Sim 部署

本目录保存 NVIDIA 官方 Ridgeback＋UR5 完整本地资产、独立底层控制器、启动入口与真实 PhysX 验收数据。当前范围为整机运动和底盘／机械臂协调；尚未接入避障 Agent。

## 启动

```bash
cd /home/w84384651/Documents/.RoboCine-Jev/ros2_ws/jev_agent_wbc/Robot_Ridgeback_UR5
./run.sh
./run.sh --headless --output results/my_acceptance
# 在有图形显示会话的终端：
./run.sh --gui --output results/my_gui_acceptance
```

默认无界面，输出目录自动按启动时间命名。指定目录必须尚不存在。启动后运行九组运动测试和 20 秒全身任务，末尾停车并退出；退出码为 0 表示物理验收通过，失败时返回非零并保留日志／测量文件。所有新运行固定 `pace=1`、100 Hz PhysX；计时结果单独报告。

本机复用已安装的 **Python 3.11／Isaac Sim 5.1.0.0**：`/home/w84384651/miniconda3/envs/env_isaaclab/bin/python`。启动器只读使用该运行时，不更改其包；机器人模型与控制代码全部在本目录。可用 `ISAAC_PYTHON=/另一个兼容环境/bin/python ./run.sh` 指定相同版本运行时。精确 Python 包版本在 `requirements-runtime.txt`，实际运行环境与源码 SHA256 在每次输出的 `provenance.json`。运行不依赖 ROS 2、Isaac Lab、训练服务或外网模型加载。

## 资产与依赖

来源为 NVIDIA Isaac Sim **5.1** 官方资产：

- `assets/Isaac/Robots/Clearpath/RidgebackUr/ridgeback_ur5.usd`
- 对应 robot schema、Grid 场景及三张纹理，共 6 个文件均已下载到本地。
- 材质 `OmniPBR.mdl` 由已安装 Isaac Sim 自带，不是远程依赖。

`ASSET_MANIFEST.json` 记录每个文件的准确 URL、大小和 SHA256。官方资产目录声明许可证为 BSD-3：<https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_robots.html>。重取／逐层检查资产：

```bash
/home/w84384651/miniconda3/envs/env_isaaclab/bin/python scripts/fetch_assets.py
```

已有本地文件时会直接检查，不下载。该脚本通过原生 USD 库解析全部 USD 资产路径，无需启动 Kit 或占用 GPU。大资产及运行缓存在 `.gitignore` 中；资产仍保留在本地。`COLCON_IGNORE` 防止 ROS 2 工作区递归误发现本目录。

## 机器人与控制方式

官方模型是 **9 个受控自由度**：平面底盘 x、y、yaw，加 UR5 六轴。模型原生使用两个 PhysX 移动副和一个旋转副驱动底盘；轮子是固定在机身上的碰撞圆柱，当前资产不包含可旋转轮关节、麦克纳姆轮滚子与相应轮地摩擦动力学。垂直移动、roll、pitch 由原生关节结构限制。因此，本部署验证的是官方平面移动操作模型的 PhysX 动态执行与机械臂协调，不包含轮地打滑或底盘竖直平衡验证。

启动时将官方模型固定根锚点的 `(-20,-20)` 场景偏移复位到原点，保留 `FixedJoint.localPos0.z=0.025930123m` 元数据。**该关节局部锚点不会将 PhysX 固定基座整体抬高，实测机身世界高度为 `z=0m`。** 对米制模型显式设定 `contactOffset=1mm`、`restOffset=0`，并从真实世界坐标碰撞壳计算地面高度：四个 wheel 圆柱最低点约 `-0.025900016m`；地面置于该最低面减去两个接触 offset 再减 `1mm`，约 **`-0.028900016m`**。这使垂向固定的原生平面模型保留约 **3mm** 碰撞壳间隙，可视轮底与地面间隙约 **2.73mm**，不产生冗余地面法向约束。接触几何全部保留。该间隙和垂向固定是模型假设，不代表轮子实际支撑和滚动。精确值、实际机身高度及全部壳体包围盒随每次输出保存。

初始关节姿态仅在 reset 后设定一次。随后全部动作通过 PhysX articulation 的关节驱动施加，没有逐帧设置机身／连杆位姿。官方源 USD 文件保持原始下载内容，启动层修改保存在实测场景文件中。

`controllers/whole_body.py` 实现加权阻尼最小二乘全身速度控制：从 PhysX 读取末端六维 Jacobian 和当前末端位姿，在同一次求解中分配底盘三轴与 UR5 六轴速度，并通过零空间关节姿态项、速度上限和位置积分限制保持连续。底盘与手臂共同跟踪世界坐标下末端位置和姿态，输出再由带实际力矩上限的关节 PD 驱动执行。参考坐标约定：米、弧度、右手系；记录四元数为 `w,x,y,z`。当前末端定义为 `ur_arm_wrist_3_link` 原点。

## 验收与实测文件

最终验收目录为 `results/acceptance_03/`。测试包含前／后／左／右、东北／西南斜行、顺／逆时针转向和停车，各 5 秒；平移速度 0.15 m/s，斜行每轴 0.10 m/s，转向 0.20 rad/s，每段首尾有 0.5 秒加减速。全身测试持续 20 秒，末端同时改变三维位置及三维姿态，底盘和机械臂均需产生非零运动才可通过。

2026-10-19 在本机 RTX 5090／Isaac Sim 5.1 上完整执行 **71 秒／7100 个物理步**，最终 **全部通过**：

| 测量 | 实际结果 |
|---|---:|
| 九方向／停车测试 | 9/9 通过 |
| 单段平移位移最大误差 | 6.075 mm |
| 单段 yaw 最大误差 | 0.001597 rad |
| WBC 末端位置 RMSE／最大值 | 0.857 / 3.568 mm |
| WBC 末端姿态 RMSE／最大值 | 0.001267 / 0.002105 rad |
| WBC 底盘 XY 跨度范数 | 0.566 m |
| WBC UR5 各关节跨度范数 | 0.901 rad |
| 最终底盘速度／yaw 角速度 | 0.000554 m/s / 0.000295 rad/s |
| 停车末秒末端坐标最大变化 | 3.04 µm |
| 物理时长／实测运行时间 | 71.000 / 71.0001 s |

`contact_probe_01` 额外保留纠正接触配置后的短诊断：两秒前进实测 `0.29999946m`；reset 后与 settle 后均核对机身实际高度，碰撞壳与地面间隙为 `0.003000m`。`artifacts_audit.json` 验证 7100 个连续 100 Hz 时间戳、源码／资产哈希一致、场景两个 USD 引用均为本地文件，以及 960×720 PNG 可以解码。

- `acceptance.json`：每组方向的实际位移、平移／转向误差、停车、WBC 误差与计时。
- `configuration.json`：真实关节顺序、Jacobians 维度、PD 参数、实际力矩限值及物理频率。
- `measured.csv`、`samples.jsonl`：100 Hz 实际关节、机身及末端状态；JSONL 另含参考末端六维姿态。
- `tested_scene.usda`：本次验收的完整场景配置，引用本地资产。
- `diagnostic.png`：物理测量结束后由同一 Isaac 场景渲染的静态诊断图，不参与控制或计时。
- `provenance.json`：用于本次执行的代码、模型清单哈希和依赖版本。
- `artifacts_audit.json`：最终交付文件的完整性与本地引用审计。

停车判定分别使用底盘 m/s、yaw rad/s 和最后一秒实际关节／末端位姿变化，避免混合不同单位。TGS 在静止承重关节可能报告小的非零 `qd`，因此保留报告速度和实际位姿变化两组原始证据。

`results/inspect_01/` 是早期 API／静止检查；`results/acceptance_01/` 是地面较低时的首次调试结果；`results/acceptance_02/` 保留地面为 `z=0` 且使用默认接触参数时的失败：固定基座的轮形碰撞壳与地面穿透约 25.9mm，冗余接触使底盘静摩擦锁死。`results/collision_geometry_audit.json` 保存排查碰撞壳的几何证据，早期推导遗漏了固定基座实际高度，已用真实测量纠正。最终结论以按碰撞壳与接触 offset 计算地面的 `acceptance_03` 为准。
