# 五款机器人本地部署与验证记录

2026-10-20 五款机器人的新 WBC 选型检查、适配进展和避障 Agent 接入门槛见
[WBC_INTEGRATION.md](WBC_INTEGRATION.md)。下文的历史运动通过不代表新框架或避障已验收。

本轮已按用户最终范围完成并结束优化：Husky、Ridgeback 使用 `run_ocs2.sh`，TIAGo 使用 `run_giskard.sh`，三款保留新控制器的运动、相机及严格停车通过结果。G1、Go2 使用各自的 `run_basic_wbc.sh`，以冻结配置完成九项任务和独立基础层验证，保留精密速度/停车未全部通过的事实。

当前分层状态与证据见 `ROBOT_DEPLOYMENT.json` 的 `wbc_integration`、各 `WBC_READINESS.json` 及 [WBC_INTEGRATION.md](WBC_INTEGRATION.md)。`--require-basic-control-ready` 检查本轮目标；原精密控制门禁和避障门禁不因基础层通过而升级。当前未接入避障 Agent，环境/传感器覆盖仍未认证。

2026-10-20 另增 [Husky＋UR5 与单臂 TIAGo](MOBILE_MANIPULATORS.md)，分别位于
`Robot_Husky_UR5`、`Robot_TIAGo`。两款采用真实轮驱动、独立机械臂与 Camera WBC，
各配一个末端相机和两个参数一致的 LiDAR。新增平台的运动、有限姿态与感知结果独立记录，
尚不满足全连续姿态无遮挡与可靠避障保证。下文保留原三款机器人 2026-10-19 的验收范围。

部署日期：2026-10-19。范围是本地模型、底层控制器及 Isaac Sim 物理仿真验证。
本轮不接入避障 Agent，不进行相机轨迹接口迁移。

三种主平台均已完成下述实测验收。机器可读入口与结果哈希见
[ROBOT_DEPLOYMENT.json](ROBOT_DEPLOYMENT.json)。

## 目录与版本

| 目录 | 模型与控制器 | 版本选择 |
| --- | --- | --- |
| `Robot_G1` | Unitree G1、NVIDIA WBC-AGILE | AGILE 官方 `v1.2`，匹配 Isaac Sim 5.1 / Isaac Lab 2.3.2 |
| `Robot_Go2` | Go2 + Piper、LeggedManip_Lab；另保留 Go2 + D1 参考项目 | 分别固定源码 commit；不混用两种机械臂的策略 |
| `Robot_Ridgeback_UR5` | NVIDIA Ridgeback + UR5 整合 USD、全身速度控制器 | 官方 Isaac 资产 5.1 |

各目录包含 `COLCON_IGNORE`，供独立运行，避免第三方包改变现有 ROS 2 工作区的构建。
Python 增补依赖安装到机器人自己的虚拟环境；现有
`/home/w84384651/miniconda3/envs/env_isaaclab` 只供读取，不在其中安装或升级包。

## 验证约定

验收依据实际 PhysX 状态，记录前、后、左、右、斜向、正反转向与停止。
足式机器人使用与预训练策略匹配的物理步长、动作频率和关节顺序。
Go2 与 Ridgeback 的协调验证需同时包含移动机构和机械臂运动，并记录末端误差。
不在运动过程中通过直接改写根部位姿模拟行走，不把模型加载或零动作测试算作稳定运动通过。

Ridgeback 官方模型使用原生 x/y/yaw 平面关节，轮子为外观几何；该资产上的测试
不能验证麦克纳姆轮的接触与摩擦或底盘垂向平衡。详细边界与测量见对应目录。

## 已保存的实测

| 平台 | 当前测量结果 | 证据 |
| --- | --- | --- |
| G1 | 65 秒连续运行、9 段通过，无中途重置或跌倒；最低根部高度 0.707 m，最大横滚/俯仰 0.059/0.084 rad | [验收 JSON](Robot_G1/results/acceptance_01/summary.json)、[原始轨迹](Robot_G1/results/acceptance_01/trajectory.jsonl) |
| Go2 + Piper | 10 个独立重置工况、共 80 秒 / 4000 控制步全部通过；协调段末端位置 RMSE 0.02186 m、姿态 RMSE 0.01815 rad | [验收 JSON](Robot_Go2/results/validation_03/summary.json)、[位姿独立复算](Robot_Go2/results/validation_03/independent_pose_audit.json)、[图像](Robot_Go2/results/validation_03/diagnostic.png) |
| Ridgeback + UR5 | 71 秒连续测试、9 组方向/停止及 20 秒全身协调均通过；末端位置 RMSE 0.000857 m、姿态 RMSE 0.001267 rad | [验收 JSON](Robot_Ridgeback_UR5/results/acceptance_03/acceptance.json)、[工件审计](Robot_Ridgeback_UR5/results/acceptance_03/artifacts_audit.json)、[图像](Robot_Ridgeback_UR5/results/acceptance_03/diagnostic.png) |

G1 速度跟踪存在偏差：前后方向指令 ±0.30 m/s 时实测均值为 +0.277/−0.269 m/s；
侧向指令 ±0.25 m/s 时为 +0.140/−0.135 m/s。通过的是所记录平地工况下的稳定方向运动，
不代表精确速度伺服或复杂地形能力。

Go2 协调段指令 `(vx, vy, wz)=(0.20, 0.10, 0.10)`，实测均值
`(0.2050, 0.0803, 0.0889)`，单位为 m/s、m/s、rad/s。
该段机械臂最大单关节幅值 0.439 rad、腿部 0.378 rad；末端采用上游 mixed-frame 目标，
不代表已实现通用世界坐标系相机 SE(3) 轨迹接口。所有工况稳定性通过；协调段最低底座
高度 0.2733 m、最大横滚/俯仰绝对值 0.2947 rad。源码与输出哈希一致，4000 帧误差
另用 SciPy 从记录的实际末端、`link0` 世界位姿独立复算通过。

Ridgeback 的最终场景保留所有碰撞壳，以实际碰撞几何世界高度设置地面：
碰撞壳最低点约 −0.025900 m，地面约 −0.028900 m，两侧接触偏移各 0.001 m，
总几何间隙 0.003 m。底盘由原生平面关节承重和驱动，底盘与 UR5 共同通过测得的
6×9 Jacobian 做带速度限幅的全身运动分配；最终协调段底盘 XY 跨度 0.566 m。
这是官方平面底盘资产上的物理关节控制验收，不能据此推断真实轮地接触与摩擦性能。

G1 补齐全部声明依赖后，[最终环境短测](Robot_G1/results/final_env_smoke_01/summary.json)
再次通过并正常退出。依赖检查通过；Kit 实际加载 Sim 自带 Warp 1.8.2，电机源代码、
Torch 2.7 / NumPy 1.26 与完整 65 秒验收一致。上游声明之间不兼容的依赖版本通过各目录
保存的元数据补丁处理，未修改官方策略、电机或物理控制算法；详细补丁与锁定版本见各目录。

## 运行入口

在相应机器人目录运行 `./run.sh` 可启动无界面验收；添加 `--gui` 显示 Isaac Sim。
使用 `--output results/<新的名称>` 保存独立结果，具体参数见各目录 README。
三套 Kit/GPU 测试按顺序运行。控制环实时比按各自测量区间统计，不含 Kit 启动与资产导入。

```bash
cd /home/w84384651/Documents/.RoboCine-Jev/ros2_ws/jev_agent_wbc/Robot_G1
./run.sh --output results/my_g1_run
```

Go2 与 Ridgeback 同样使用各自目录内的 `run.sh`。默认无需 ROS 2 桥接或任何 Agent API。
源码、模型、策略、增补依赖均分别保存在对应 `Robot_*` 目录中；大文件与虚拟环境
留在本地并由各目录 `.gitignore` 排除，来源、补丁和重建脚本随代码保留。

Go2 的 D1 参考工程已完成本地 URDF 合并和 Isaac Sim USD 导入，保留全部网格及 ROS 2
控制源码。原始 D1 URDF 的机械臂/手指关节 effort、velocity 为零，不能直接据其预设
步态宣称整机稳定或 WBC 通过；本次 Go2 的动力学验收明确采用 Piper 组合。

这些结论只覆盖已记录的平地、无障碍工况。相机任务接口、通用世界坐标轨迹、避障 Agent
和复杂地形不属于本次验收。Go2 的 D1 工程为单独的模型/控制源码参考，不能将 Piper 的
验收结果移用于不同机械臂或其他模型变体。
