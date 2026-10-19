# Go2＋机械臂本地部署

本目录保存用户指定的两个官方仓库及其模型、控制器、策略。全身控制主平台选择 **Go2＋Agilex Piper**，因为 LeggedManip_Lab 同时提供匹配的 USD、WBC 配置和已导出的 TorchScript 策略。Go2＋D1 保留为独立参考平台，不能使用 Piper 的策略。

## 运行

在本目录执行：

```bash
./run.sh --output results/my_validation
./run.sh --gui --snapshot --cases forward,coordinated --output results/my_gui_run
```

默认无界面，在 Isaac Sim 5.1 中使用 CPU PhysX 和 CPU 策略推理，以降低显存占用。物理步长为训练配置的 0.005 秒，控制周期 0.020 秒。`pace=1`，每一步只补足实际时间，不额外减速；结果分别记录仿真时间和实际耗时。使用 `--device cuda:0` 前应确认本地 Torch 构建支持 GPU。

默认验收前进、后退、左移、右移、左右斜行、正负转向、停止及移动中末端目标变化，共 10 组，每组 8 秒。每组独立重置，2 秒稳定后发出指令，3 秒之后统计误差。机器人为自由浮动动力学模型；执行过程中不写底座位姿、不强制底座速度、不施加支撑力，也不使用自动重置隐藏跌倒。

这里的停止场景是零速度站立；10 组之间显式重置，因此不代表连续无重置切换方向的耐久验收。

输出包括 `summary.json`、`measurements.jsonl`、逐场景 `progress.json`。`--snapshot` 额外保存模型场景 USD 和一张最终状态诊断 PNG；图像不作为测量或控制输入。

验收阈值：完整执行规定步数，所有状态、姿态、动作和末端误差均有限；最低底座高度大于 0.16 米，最大 roll/pitch 小于 0.65 弧度；平均平移速度误差小于 0.15 米/秒，平均角速度误差小于 0.15 弧度/秒，非零指令对应的实际速度方向正确。静止场景更严格，平移速度不大于 0.08 米/秒，角速度不大于 0.12 弧度/秒。协调场景还要求腿部和机械臂都产生关节运动、末端位置 RMSE 小于 0.15 米、姿态 RMSE 小于 0.6 弧度。阈值及原始数据均保存在脚本和结果中；任一场景失败则进程返回非零退出码。

## 模型和控制接口

| 项目 | 本地目录 | 用途 |
|---|---|---|
| [LeggedManip_Lab](https://github.com/zzzJie-Robot/LeggedManip_Lab) | `vendor/LeggedManip_Lab` | Go2-Piper / Go2-ARX5 的 USD、训练环境、控制器及策略；本次选 Piper WBC |
| [Isaac-sim-go2-with-arm](https://github.com/nayon007/Isaac-sim-go2-with-arm) | `vendor/Isaac-sim-go2-with-arm` | Go2 和 D1 分体 URDF、全部网格、ROS 2 crawl/trot 控制源码 |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | `vendor/IsaacLab` | 与 Sim 5.1 / RSL 5.0.1 兼容的固定 main 提交 |

精确提交号、许可证文件和策略 SHA256 见 `SOURCES.json`。当前固定的 Lab main 提交支持 Python 3.11 / Isaac Sim 5.1 / 新 RSL 5 API，虽然根 `VERSION` 仍写作 2.3.2，但它与同名旧 release tag 不是同一源码。

Piper WBC 策略输入为 210 维（7 组特征、3 帧历史），输出为 18 维（12 腿＋6 臂）。策略动作按官方配置缩放 0.25 并加默认关节角，通过官方 PD 执行器控制，不需要 ROS 2 桥接。末端目标采用上游定义的 mixed frame：x/y 和姿态相对于 `link0`，z 为世界高度；它不是任意世界坐标系下的相机 6D 轨迹接口。本次未接入避障 Agent。

## D1 参考模型

上游没有提供已合并的整机 USD，只提供分体 URDF 和合并说明。运行以下命令将两份 URDF 合并，并将所有网格资源解析为本地路径：

```bash
.venv/bin/python scripts/prepare_d1.py
./import_d1.sh --headless
```

本地安装位置为 Go2 `base` 上方 `[0.05, 0, 0.10]` 米，是本次明确记录的装配选择。原始仓库文件保持完整。注意：D1 URDF 实际包含 6 个转动臂关节和 2 个移动手指关节，Go2＋D1 合计 20 个活动关节；上游 ROS 控制器只命令其中 19 个，并将机械臂固定为零。其预设步态没有完整的多方向速度接口。该源码不能作为本次 Piper 全身协调实测的替代证据。

原始 D1 URDF 的全部臂／手指关节还将 effort 和 velocity 上限设为 0，正式动力学控制前需要使用可信执行器规格补全，不能随意替换成 Piper 参数。D1 的 Isaac 动力学控制尚未验收，Piper 原生 USD 与 WBC 验收独立完成。

D1 已成功导入为 `assets/generated/go2_d1/go2_d1.usd`，包括约 28 MB 的本地 USD 配置层和全部 20 个活动关节，底座未固定；`import_report.json` 保存关节清单及 URDF／USD SHA。转换使用本机 Isaac Sim 5.1 自带的原生 URDF importer 2.4.30。Lab main 的通用转换器强制下载 2.4.31，首次尝试因 Nucleus 扩展获取阻塞；`import_d1.sh` 现直接调用本机官方原生接口，不需要该下载。成功日志为 `results/import_d1_02.log`。原始无视觉几何的 Head_upper／Head_lower 在转换时产生两个空 visual 引用，现已仅在生成的 base 层清理经完整 stage 确认不存在的两个目标；合法内部引用、原始碰撞和物理层均未修改，清理清单及各层 SHA 保存在导入报告中。不据此声称完成 D1 动力学验收。

## 环境隔离与重建

```bash
./bootstrap.sh
```

默认复用 `/home/w84384651/miniconda3/envs/env_isaaclab/bin/python` 所在的 Isaac Sim 5.1，只在本目录的 `.venv` 安装附加依赖，不修改该共享 conda 环境；可用 `ISAAC_PYTHON=/path/to/python` 指定其他已有 Sim 5.1 环境。`constraints.txt` 固定 Torch、NumPy、TensorDict 和 Pin，避免安装器解析到不兼容的新大版本。完整安装日志保存在 `results/`。本目录的 `COLCON_IGNORE` 防止 vendor 中的 ROS/Lab 包改变已有工作区构建。

`isaaclab_sim51_metadata.patch` 仅将 Lab 的 Starlette 依赖从 0.49.1 改为 0.45.3，以兼容 Isaac Sim 5.1 固定的 FastAPI 0.115.7。补丁不修改物理、模型、观测和策略；本次不启动 Web 直播服务。补丁由 bootstrap 自动应用。地板和天空均为本地资源，验收不依赖 Nucleus 云端环境。

## 已完成的实测

`results/validation_03/summary.json`：10／10 场景通过，4000／4000 控制帧完整且有限；80.0 秒仿真耗时 80.85 秒，实时因子 0.9895，独立 `timing_pass=true`。

| 场景 | 指令 vx, vy, wz | 实测平均 vx, vy, wz |
|---|---|---|
| 前进 | 0.20, 0, 0 | 0.184, 0.004, -0.007 |
| 后退 | -0.20, 0, 0 | -0.129, -0.002, 0.001 |
| 左移 | 0, 0.20, 0 | 0.032, 0.160, -0.014 |
| 右移 | 0, -0.20, 0 | 0.027, -0.168, 0.012 |
| 左斜 | 0.20, 0.20, 0 | 0.219, 0.171, -0.005 |
| 右斜 | 0.20, -0.20, 0 | 0.217, -0.174, -0.004 |
| 左转 | 0, 0, 0.20 | 0.011, -0.022, 0.141 |
| 右转 | 0, 0, -0.20 | 0.012, 0.012, -0.172 |
| 移动＋机械臂 | 0.20, 0.10, 0.10 | 0.205, 0.080, 0.089 |

移动＋机械臂场景中，腿部最大关节活动幅度 0.378 rad、机械臂 0.439 rad，末端 mixed-frame 位置 RMSE 0.0219 m、姿态 RMSE 0.0181 rad，机器人最低高度 0.273 m，最大绝对 roll/pitch 为 0.295 rad。验收是平地标称工况下的 WBC 初验，不代表任意相机轨迹的精确跟踪、复杂地形或碰撞避障能力。

`measurements.jsonl` 保存每帧实际底座状态、18 个关节位置、动作、末端目标、末端和 link0 的世界位姿。独立 SciPy 复算 4000 帧末端误差，与在线指标最大差仅 6.9e-8 m／5.3e-7 rad；可运行：

```bash
.venv/bin/python scripts/audit_measurements.py results/validation_03
```

该目录还保存 `validator_source.py`、`arguments.json`、源码 SHA、模型／策略来源和 `diagnostic.png`／`diagnostic.usd`。结果目录已存在时拒绝覆盖。启动器会检查本轮 `run_complete`，不能将 Kit 初始化错误退出 0 当成通过。

当前 `scripts/validate_wbc.py` 与该轮冻结源码完全相同，SHA256 为 `c3280c6e34d0f3438913bd73377212bc32533ad5cc4f16a3c0250920afe181d6`。`asset_closure.json` 检查了 Piper 的全部 5 个本地 USD 层；无外部云模型资产，OmniPBR 材质来自 Isaac Sim 本地内置文件并记录 SHA，未解析依赖为空。`pip check` 已通过。

调试历史 `validation_01/02` 和 `reset_diagnostic_01/02` 均不作为已完成验收。确切根因是验收脚本在 inference_mode 运行物理后，却在模式外做第二次 reset，引发 PyTorch inference tensor 原地写入异常；随后 Kit 关停回调循环掩盖了错误。现已统一 reset 的推理上下文，先保存异常，再禁用仅用于关停的等待回调。最初的纹理加载猜测已被后续调用栈和实际异常排除。物理步长、执行器、策略、动作及观测未因此改变。
