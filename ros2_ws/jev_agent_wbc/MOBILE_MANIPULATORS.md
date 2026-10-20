# Husky＋UR5 与单臂 TIAGo（2026-10-20）

新增模型、传感器配置、固定来源、转换记录和实测文件分别位于
[Robot_Husky_UR5](Robot_Husky_UR5/README.md)、[Robot_TIAGo](Robot_TIAGo/README.md)。
共享入口与控制接口见 [运行说明](scripts/mobile_manipulator/README.md)。
原有 G1、Go2、Ridgeback 的模型、传感器和历史结果没有修改。

| 项目 | Husky A200＋UR5 | TIAGo 单臂差速版 |
| --- | --- | --- |
| 实际运动机构 | 四个滚动轮，滑移转向 | 两个驱动轮，被动脚轮支撑近似 |
| 独立机械臂控制 | UR5 六轴 | 七轴机械臂与升降躯干 |
| 相机 WBC 控制变量 | `v, ω, dq1..dq6`，8 维 | `v, ω, dh, dq1..dq7`，10 维 |
| Camera 父刚体 | `ur5_wrist_3_link` | `gripper_link` |
| Camera 平移（父坐标，m） | `[-0.04, 0, 0.34]` | `[0.16, -0.04, -0.24]` |
| LiDAR 数量 | 一个主扫描器＋一个辅助 | 一个主扫描器＋一个辅助 |
| 物理环境 | Isaac Sim 5.1 / PhysX，100 Hz，`pace=1` | 同左 |

底盘不具备直接横移自由度，左右和对角目标通过实测朝向反馈转向后行进。
机械臂及躯干通过有官方力矩/力上限的物理驱动执行；Husky 轮电机参数为未做实机标定的
仿真配置。相机的完整安装外参进入 FK 和 Jacobian。WBC 是带位置、速度、加速度和轮速
约束的局部 QP，不依赖 Jev 或其他机器人控制器，不包含完整自碰撞距离约束或环境路径规划。
模型只在初始化时安放，运动过程中不重写机身和连杆位姿。

Husky 的社区模型包含错误质量、惯量和显式几何包围盒，以及冗余夹爪闭链；
修复以固定版本随附 URDF 为依据，夹爪固定为末端工具，六个机械臂关节保留。
TIAGo 来自 PAL 官方单臂描述，保留七轴与 0–0.35 m 躯干行程；处理了导入器的网格格式、
零质量空框架、驱动单位和脚轮摩擦组合问题。官方原始文件与所有失败记录均保留，
派生 URDF 对 37 个共同链接、273 个构型的运动学一致性误差不超过 `4.44e-16`。
具体物理近似、来源提交和文件哈希见各自 README/清单。

两款最终完整运动验收及独立审计均通过。相机位置误差由保存的实际变换独立重算，
停车同时检查机身/相机的平移、旋转以及机械臂/躯干位置变化。

| 最终运动实测 | Husky | TIAGo |
| --- | --- | --- |
| 六方向、独立机械臂、Camera WBC、停车 | 全部通过 | 全部通过 |
| 连续 100 Hz 实测样本 | 11,992 | 9,131 |
| WBC 相机位置 RMSE / 最大值 | 8.448 / 16.989 mm | 2.040 / 5.949 mm |
| WBC 相机姿态 RMSE | 0.009678 rad | 0.003220 rad |
| WBC 底盘运动跨度 | 0.11243 m | 0.12580 m |
| 独立机械臂时底盘漂移 | 1.268 mm | 0.086 mm |
| 仿真时长 / 实测墙钟 | 119.92 / 185.452 s | 91.31 / 93.575 s |
| 独立计时判据 | 未通过 | 通过 |

Husky：[实测](Robot_Husky_UR5/results/acceptance_05/acceptance.json)、
[独立审计](Robot_Husky_UR5/results/acceptance_05/motion_audit.json)、
[实际概览图](Robot_Husky_UR5/results/acceptance_05/overview.png)。
TIAGo：[实测](Robot_TIAGo/results/acceptance_final/acceptance.json)、
[独立审计](Robot_TIAGo/results/acceptance_final/motion_audit.json)、
[实际概览图](Robot_TIAGo/results/acceptance_final/overview.png)。
扫描、并行测试和记录开销计入墙钟；没有修改物理时间来获得计时通过。
Husky 后增的软限位代价在上述 1,600 个 WBC 实测构型中均为零，
最小限位距离 1.017 rad，大于软区 0.25 rad，见
[状态复核](Robot_Husky_UR5/results/soft_margin_audit.json)。当前共享控制器还另行通过了
[20 秒 WBC/停车实测](Robot_Husky_UR5/results/wbc_soft_final/diagnostic_audit.json)，
不把该短诊断视为重新完成全部方向测试。
TIAGo 则实际使用该代价，让底盘在机械臂接近限位前分担运动。
硬位置、速度、加速度与提前制动约束保持有效，不可行时停止发送运动候选。

传感器均参与真实 Isaac 场景：Camera 使用实际 RTX RGB，LiDAR 使用 PhysX 最近命中射线。
同一机器人两个 LiDAR 的采样、视场和量程参数完全相同，保留本体命中与近距离盲区。
LiDAR 没有模拟 RTX 波形、材质、扫描时序或噪声；原型传感器也未增加支架、线缆、质量和惯量。

姿态检查保持原始关节范围：Husky 为 12 个单轴端点＋64 个 Sobol 组合＋默认姿态，
共 77 个；TIAGo 包含躯干，16 个端点＋64 个组合＋默认姿态，共 81 个。
每个姿态实际渲染 RGB/本体掩码，并测试 0.7 m 和 1.5 m 两圈、五个高度的 12 cm 障碍物。
固定基座且关闭重力的几何检查与连续轮地运动分别记录；有限采样不证明连续全空间。

| 传感器实测 | Husky | TIAGo |
| --- | --- | --- |
| 原机器人无非相邻重叠的姿态样本 | 31/77 | 55/81 |
| 该子集中相机无遮挡且外壳无干涉 | 31/31 | 51/55 |
| 全部姿态、全部测试障碍的返回 | 27,461/27,720（99.07%） | 29,115/29,160（99.85%） |
| LiDAR 安装原点与本体重叠 | 0 | 0 |
| 全姿态/避障充分性验收 | 未通过 | 未通过 |

Husky 扫描证据见 [完整扫描](Robot_Husky_UR5/results/coverage_candidate_01/summary.json)、
[最终相机](Robot_Husky_UR5/results/camera_final_01/summary.json)。前者与最终相机配置的
机器人几何、扫描器参数和安装位置相同，相机节点没有碰撞几何；两次试验分开保留。
TIAGo 的最终配置见 [完整扫描与相机](Robot_TIAGo/results/coverage_best/summary.json)。
原始 NPZ、逐姿态 RGB、掩码及独立复算结果保存在对应目录。

可直接在 Isaac Sim 打开的传感器组装层为
`Robot_Husky_UR5/assets/husky_with_sensors.usda`、
`Robot_TIAGo/assets/tiago_with_sensors.usda`；它们引用版本固定的机器人几何并附带
实际 Camera 和两个扫描坐标系。LiDAR 最近命中采集仍由 `run.sh` 实现。
重现驱动行为须使用运行入口、冻结配置和 `configuration.json`；只打开几何场景不会
自动恢复通过物理接口设置的全部运行时增益和力矩上限。

漏检并非全是扫描密度不足：部分障碍被前方已经检测到的其他障碍遮挡，部分受到真实本体遮挡，
Husky 还有最小量程内的近场返回。独立审计保留具体原因，不能把被遮挡区域当作自由空间。
TIAGo 仍有四个机器人本体无重叠的样本存在相机遮挡/外壳干涉：
`arm_5_joint_low`、`sobol_007`、`sobol_030`、`sobol_048`，可查看原始图像和关节状态复现。
未通过样本没有删除，关节范围没有缩小。

因此当前交付是两套可运行的物理模型、控制与可复核的验证流程。
“任意原始关节姿态相机始终无遮挡”“两个 LiDAR 对所有姿态足以保证避障”仍未实现，
不能将运动通过、单帧清晰或高比例目标检测当作这些保证。新平台尚未连接现有避障 Agent；
运行任意外部相机轨迹也没有连续碰撞/可见性安全证书。

运动结果读取各自最新 `acceptance.json` 与独立 `audit_motion.json`，
传感器结果读取 `summary.json` 和原始证据审计，`timing_pass` 单独判定。
父进程对 Kit 退出码做二次核验，缺失产物、未完成、错误或失败均不能因为 Kit 退出 0 而通过。
原始图像是传感器/诊断证据，没有将未经过 Representation 的视频作为整机影片交付。

相关回归检查最终为 **69 passed、3 skipped**。覆盖非完整约束与相机 Jacobian、
持续推向所有关节上下限、限位附近跟踪扰动、制动/反向、外部轨迹校验、
缺失结果拒绝、旋转停车和原始扫描/图像一致性。3 项需要直接加载 USD 的单元测试
在普通 Python 中跳过；实际 Isaac/RTX/PhysX 验证使用上列独立实测文件，不以跳过项充当通过。
