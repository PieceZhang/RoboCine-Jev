# Ridgeback + UR5：真实 OCS2 / Isaac Sim 闭环

`run_ocs2.sh` 已把真正的 `GaussNewtonDDP_MPC`/SLQ 接入 Isaac Sim 实测物理状态；历史 `run.sh` 保留原控制器。当前单一地面场景的完整 **FULL02通过八方向（保留body vy）、独立机械臂、原21s相机6D Task、每段全9DOF严格停车、最终Task容差、pace1及真实RGB／光学方向核验**。四份根级独立审计均通过，控制／轨迹／公共接口三项gate为true，环境与传感器避障gate仍为false。机器可读状态在 [WBC_READINESS.json](WBC_READINESS.json)；旧控制器、CPU预测及旧场景的源码等价证明不代替这次实际物理重跑。

使用 [OCS2 官方源码](https://github.com/leggedrobotics/ocs2) `ros2` commit `88ce706c17fcc4591b6c24c87a4a85ff6ed9e92d`。固定 URL/逐文件哈希与许可证见 `ocs2/UPSTREAM_MANIFEST.json`、`OCS2_LICENSE.txt`。隔离构建 `thirdparty/core/oc/mpc/qp_solver/ddp` 六包，未升级已有 Isaac Python 环境、未修改上游源码。完整官方移动操作示例的 Pinocchio/HPP-FCL 包未构建；自定义 USD→POE 运动学及真实 collider 保守凸包属于本项目模型，优化器确为原生 OCS2。[官方示例说明](https://leggedrobotics.github.io/ocs2/robotic_examples.html) 用于对照模型范围。

真实保留 vx_body/vy_body/yaw 全向输入，映射到原资产的世界 x/y 移动副和 yaw 旋转副。该官方资产固定高度/roll/pitch，轮子没有受驱滚动或 mecanum 滚子动力学；本次不能证明真实 mecanum 底盘性能。 原始关节位置界保持不变。模型的9维位置状态来自每个100Hz PhysX步的实测完整底盘SE(3)及六关节；预测扩展为17/18维，尾部明确定义为**上一条实际下发的速度参考**，输入是参考变化率，不是把理想速度冒充物理测量，也不是完整二阶接触动力学。原始实际 qd/base velocity 另行记录并执行 overspeed、原限位与 qd·dt+qd²/(2a) 刹停余量检查。任何实测限位/速度/余量违例都进入正式总失败，`.015rad` emergency guard 不会把原限位失败改成通过。

固定高度的非滚动轮不能与地面形成冗余支撑约束：沿用旧已验证入口的精确 collision minZ，地面位于 minZ−3mm，机器人和地面 collider contactOffset 均为1mm/restOffset为0。首次新入口漏掉该双侧 offset 导致启动冲击，失败完整保留在 `results/ocs2_camera_01`；恢复后 camera02 与全套 acceptance01 均零实测安全违例。修复没有移动运行中的机器人、删除碰撞或放宽速度阈值。原64/16 solver保留。

原生 worker 为长期单进程，一次一个在途求解；1s预测时域、ODE初始步长.01s、SLQ最多5次迭代。物理始终继续前进；策略按原始测量绝对时间及墙钟双重.2s lease失效，失败时有界制动、fresh state重算，最多5次失败或2s无新策略后停止并保留失败记录。每次请求只截取原始轨迹中完整覆盖当前horizon及两端插值段的连续knots；没有重采样、移动、重定时或修改相机路径。

公共 `TimedTask` / `LiveTaskSession` 实际控制相机轨迹；world语义相机轴显式转为 optical（+Z向前、+Y向下），保持绝对时间、线性位置/最短SO(3)插值。`accepted_observations.json` 保存接纳时的实际Observation。反馈从实际碰撞GPrim缓存形状、每tick PhysX刚体SE(3)构成移动平台/相机机械臂两部分保守几何；关节、FK、Jacobian留在私有日志。只有 `timed_camera_se3` 已支持；身体区域/构型偏好明确 unsupported，`supported=false, feasible=null`，不会伪装为已证明几何不可行。外部任务仍须绑定新鲜测量Observation；当前内置轨迹是有固定输入、原始任务及哈希的诊断案例。

| 当前完整实际物理套件 FULL02 | 结果 |
|---|---:|
| 完整结果 | `results/ocs2_acceptance_02/acceptance.json` |
| 八方向及每段严格停车 | 全通过，平移误差最大3.040mm |
| 独立六关节／底盘漂移 | 各约.239–.241rad／3.35微米 |
| 相机位置 RMSE／最大值 | 3.351／15.049 mm |
| 姿态 RMSE／最大值 | 0.005573／0.012095 rad |
| 原Task逐点位置／姿态超差 | 0/2100／0/2100 |
| 最终原Task位置／姿态误差 | 3.336mm／0.009581rad |
| 非零底盘／手臂变化范数 | 0.6552m／1.1722rad |
| 物理／墙钟 | 122.00／122.053s |
| 全100Hz样本／实测安全违例 | 12200／0，包含settle前缀 |
| 全套／独立审计／原始RGB | 全通过／四项通过／通过 |

`validation/wbc_ridgeback_full02_{physical,suite,joint_limits,render}_independent_20261020.json` 从raw重算原Task、光学轴、接纳Observation ID、反馈与时序、全部物理DOF及source哈希、八方向／六关节独立动作和每段停车、原始限位与实际640×480渲染。两条平移副使用m，yaw及手臂使用rad。`ocs2/evidence/full02_pointwise_scope.json`独立重算2100个轨迹采样与原Task终点；本条轨迹全部采样点在35mm/.10rad内，不构成连续全工作空间保证。

严格停车预声明5s预算，检查末.5s全部真实DOF（包含Husky轮子、Ridgeback的m/m/rad底盘）及底盘/相机完整SE(3)：位置span<5mm、两两姿态span<.01rad、转动关节span<.01rad、移动副<2mm、最终实测底盘全线速度<.02m/s和角速度<.03rad/s；pace要求wall≤1.05×sim。不会只检查机械臂、缩短窗口或丢弃settle。独立审计脚本重新从物理日志与原始Task计算结果。

CPU理想模型闭环另存 `results/ocs2_model_home_01/summary.json`：21s完成，位置RMSE 2.907mm、姿态RMSE 0.005633rad、最长solver 0.0792s。这是模型预测结果，与上表物理精度分开。62项CPU／原生USD后端测试包含真正原生持久solver错误恢复/绝对时间、随机解析导数、凸包守卫、实测qd制动、轮式servo及不可达参考拒绝；初次失败、旧单horizon结果和CPU版文档保留在 `ocs2/evidence/`。

关节/速度/输入率在SLQ内仍为 relaxed barrier **软约束**（mu=.2、delta=.005），不是硬约束。发布前按原始硬上限检查所有策略knots；刹停代价及真实碰撞凸包距离局部线性化参与优化，全部非固定/非相邻collision pairs保留，发布前再做非线性凸包检查，物理每tick另有真实形状guard。凸包是保守外包，仍可能误拒绝；knots及100Hz检查不构成连续域全工作空间保证。未建立力矩、驱动延迟、接触/滑移的完整预测模型。驱动力/速度cap已配置并读回，实测速度单独审计；projected reaction effort不能等同隐式PD实际电机扭矩，因此公共actuation正向状态保持unknown。

环境约束与真实LiDAR避障未验证，传感器已有有限姿态/渲染证据与已知盲区仍保留；environment与sensors_avoidance gates为false，未知不会填pass。历史 `results/acceptance_03/acceptance.json（weighted damped least squares）` 仅证明旧控制器，不替代新OCS2验证。

```bash
# 当前目录内；固定来源隔离构建，不升级现环境
./ocs2/scripts/build_native.sh --mpc
# 新建输出目录；--wbc-only 仅相机诊断，默认含方向/独立臂/相机/每段严格停车
./run_ocs2.sh --wbc-only --output /absolute/new_camera_result
./run_ocs2.sh --output /absolute/new_full_result
```

原生内部单horizon入口 `ocs2/scripts/run_native.sh` 仍可复现旧CPU案例，但不会发送物理命令。新入口冻结执行源、原生binary/guard、模型/传感器配置、完整原始Task及source SHA；流式记录全部DOF、真实速度、body/camera/reference SE(3)、约束反馈和分项wall耗时，结束保存flattened tested scene。`ocs2/.cache`与大体积results被忽略，但构建脚本/固定manifest/持久CPU结果保留。

历史FULL01使用旧双地面场景。后续仅Husky分支的精确diff、Ridgeback专门化AST相等、12200条旧q/qd回放及根级独立等价审计仍完整保留在 `ocs2/evidence/branch_equivalence_*/` 与 `../validation/wbc_ridgeback_branch_equivalence*/`；它们只证明各自冻结的旧源码分支关系，不能证明更改地面后的场景。本次FULL02实际从新world完整重新执行，当前正式门禁直接绑定其冻结源及原始测量，无需把历史等价证明冒充新物理证据。

当前场景修复SDK GroundPlane的双重floor collider：仅保留原analytic Plane、绑定原.5/.5材质，visual geom关闭碰撞；robot全部shape／pair／offset／drive／caps保持。真实旧flattened scene的CPU gate验证2→1个floor、材料绑定与整个Robot Sdf spec SHA不变，62项USD／native回归通过；FULL02已实际完整复测通过，`ground_audit_before_reset.json`及`ground_audit.json`强制核对唯一floor与原材料。新的汇总按已接纳Task自身容差重算最大／RMS／逐点超差数／最终误差，原始参考及绝对时间未修改。历史 `ocs2_camera_01` 的错误接触启动冲击、原始失败及所有源快照保持不变。
