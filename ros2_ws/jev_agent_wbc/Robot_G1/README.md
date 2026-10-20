# Unitree G1 + NVIDIA WBC-AGILE 本地部署

本目录固定官方 AGILE v1.2（ad4ff6ee6b38da207592b3e71dcee4c118049f6d）、Isaac Lab v2.3.2（37ddf626871758333d6ed89cf64ad702aef127d0），复用已安装的 Isaac Sim 5.1.0.0/Python3.11。官方 v1.2 README 明确支持这一组合；较新的 AGILE v1.3 面向 Sim6/Python3.12，因此没有混装。

已下载完整官方 G1 29自由度 USD 及其相对依赖（149MB），并取得所有 v1.2 G1 预训练策略实际文件（10.1MB，非 Git LFS 指针）。源码、电机控制器、训练/评估脚本保留在 upstream。来源和每个资产的 SHA256 分别位于 config/sources.json、config/assets.manifest.json。

## 当前基础 WBC 入口

本轮按基础控制范围收尾：采用官方 **velocity-height 12腿策略 + 真正Pink 15上身关节任务层**，目标是稳定运动和任务/实测反馈接口适配。推荐入口明确固定已实测的Ki1配置、当前状态重力支撑和原关节限值；未实测的Ki4实验不作为默认。

```bash
cd /home/w84384651/Documents/.RoboCine-Jev/ros2_ws/jev_agent_wbc/Robot_G1
./run_basic_wbc.sh --output results/my_basic_run
```

默认九项包括前后、左右、双斜向、双转向和移动摄影，共92秒；每项5秒运动和5秒停车。输出保留原严格`summary.json`和逐步原始数据，基础控制的独立判定与精密速度/停车结果分开。原严格退出码可能为2，不应擅自改写为通过。验收证据、范围和剩余限制见[WBC.md](WBC.md)及[WBC_READINESS.json](WBC_READINESS.json)。

最终[九项实测](results/basic_full_validation_01/summary.json)在92.000093秒墙钟完成，[独立基础审计](../validation/wbc_g1_basic_full01_independent_20261020.json)通过、0错误，4500个活跃采样的实际形状碰撞检查均新鲜通过。原精密速度8/9、严格停车4/9通过：左转实测.1520rad/s高于.12目标，倒退、双斜向、左转和移动摄影仍有严格关节停车偏差。4次真实短暂QP失败与原`all_pass=false`完整保留；本轮据此完成基础层收尾，不再继续调参。

这里不承诺精密速度跟踪、任意末端六维轨迹或避障。公共接口接收绝对仿真时间的WORLD相机任务，返回实际相机和两部分碰撞几何观测；环境与无法认证的执行约束仍返回unknown。短暂缺帧时保持已提交的上身位置、速度目标0，每个物理子步重新计算静态重力，不复用过期运动包。

## 保留的历史行走入口

以下`run.sh`及65秒结果属于旧255输入/14输出velocity-history策略，作为历史证据保留，不能替代当前height+Pink的基础WBC验证。

```bash
cd /home/w84384651/Documents/.RoboCine-Jev/ros2_ws/jev_agent_wbc/Robot_G1
./run.sh                                      # 默认65秒多方向物理验收
./run.sh --gui --snapshot                     # 界面、诊断截图
./run.sh --seconds 4 --output results/my_smoke # 短诊断前缀，输出路径必须不存在
./run.sh --commands config/commands.json      # 自定义机身vx/vy/yaw指令序列
```

启动依赖本机NVIDIA驱动和可访问GPU的执行环境。脚本使用独立 .venv，仅只读继承原 conda 的模拟器，控制策略和PhysX用CPU，Kit用GPU。没有改动共享环境。全部声明依赖已补齐，pip check无缺失/冲突；四处上游元数据兼容调整见docs/DEPENDENCIES.md。重新取源码、资产并建立venv可运行 scripts/bootstrap.sh，或用 ISAAC_PYTHON 指定另一个已安装Sim5.1的Python解释器。

主用策略为官方 Velocity-G1-History-v0，14维动作控制双腿与腰部roll/pitch，其余关节保持默认姿态。保留原始200Hz物理与50Hz策略频率、延迟DC电机模型、255维观测和精确关节顺序。仿真期间不设置根部位姿或速度。详细先验阈值、训练与验收环境差异见 docs/ACCEPTANCE.md。

每次运行写出 metadata.json、trajectory.jsonl、summary.json 与场景USD；可选诊断PNG。stable_multidirection_pass 只有完整65秒工况、全部速度跟踪与姿态/高度门槛都通过时才为true。短诊断不能作为完整稳定行走验收。

2026-10-19 已在本机 Isaac Sim 完成完整65秒物理验收：[summary.json](results/acceptance_01/summary.json) 的 `stable_multidirection_pass=true`。3250个控制周期无中途重置、跌倒或非有限值；最低根部高度0.707m，最大roll/pitch分别0.059/0.084rad，仿真/墙钟实时比1.000。

|工况|目标 vx/vy/wz|实测均值 vx/vy/wz|
|---|---|---|
|前进|0.30 / 0 / 0|0.277 / 0.011 / 0.004|
|后退|-0.30 / 0 / 0|-0.269 / -0.002 / -0.007|
|左移|0 / 0.25 / 0|0.020 / 0.140 / 0.001|
|右移|0 / -0.25 / 0|0.022 / -0.135 / 0.008|
|斜向|0.25 / 0.20 / 0|0.239 / 0.122 / 0.003|
|左转|0 / 0 / 0.40|0.005 / -0.010 / 0.391|
|右转|0 / 0 / -0.40|0.008 / 0.013 / -0.394|
|停止|0 / 0 / 0|-0.004 / 0.003 / 0.014|

侧向实测约为目标速度的54%～56%；本次通过的是平地多方向稳定行走验收，尚未达到精确速度伺服。无需改变预训练策略即可稳定向要求的方向运动。

审计数据在 [results/acceptance_01](results/acceptance_01)：metadata保存策略/控制器/IO/指令/资产manifest/源码的SHA256和包版本；source保留本次实测原始文件；trajectory.jsonl为逐周期实际状态。初次缺h5py的启动日志smoke_01保留，补齐依赖后的smoke_02通过4秒站立。

headless体验没有active viewport，因此没有生成诊断PNG，具体见snapshot_error.txt；不宣称交付了截图。首次Kit在完成物理与保存证据后卡在收尾，已仅结束本任务进程；已定位并禁用Lab停止回调的等待渲染循环，run.sh另用独立监督器在完成/异常证据保存15秒后处理收尾问题，并校验本次新目录、完成标记、进程退出原因与summary。此更新只涉及结束流程，实测源文件快照保持原样。启动过程中unused SpaceMouse缺hid错误已补hidapi到本地venv和bootstrap；并未更改实测中的控制器。

补齐全部声明依赖后，已通过 [final_env_smoke_01](results/final_env_smoke_01) 的4秒物理复验，Kit实际仍使用内置Warp1.8.2；200步完整状态与原smoke_02逐项完全一致，且正常关闭（无监督器终止）。最终pip check为“No broken requirements found.”。

本目录暂不包含 Agent 避障适配。

## 运镜 Camera 与机身 LiDAR

当前入口安装 1 个右腕 Camera、1 个上躯干主 LiDAR 和 3 个机身辅助 LiDAR，安装变换和相机光轴均显式配置。
`--sensor-audit` 采样本体遮挡和实测 TF；`--sensor-rgb` 在运行后检查实际相机图像与前方光轴。
配置、输出及未验证范围见 [三平台传感器安装说明](../SENSOR_INSTALLATION.md)。
新增传感器尚未接通避障 Agent；历史行走验收不等于全姿态传感器覆盖验收。

## 新 WBC 适配核验（2026-10-20）

`run_wbc.sh`和早期6秒试验保留用于历史复现；当前基础入口是上面的`run_basic_wbc.sh`。此前独立16秒明显六维相机轨迹曾通过，但属于另一个冻结控制版本，不能把结果转移到本轮基础默认配置。实现、失败历史、实测接口与能力边界见[WBC.md](WBC.md)和[WBC_READINESS.json](WBC_READINESS.json)；环境约束和避障门槛保持未通过。
