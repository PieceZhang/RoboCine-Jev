# Husky + UR5：真实 OCS2 / Isaac Sim 闭环

`run_ocs2.sh` 已把真正的 `GaussNewtonDDP_MPC`/SLQ 接入 Isaac Sim 实测物理状态；历史 `run.sh` 保留原控制器。当前单一地面版本完整 **FULL03通过六方向、六关节独立控制、原21s相机Task、每段全10DOF严格停车、最终Task容差、原速度／位置界、pace1和原始RGB**，四份独立审计均通过，三项控制／轨迹／公共接口门禁为true。整体按原RMS与终点验收，40/2100帧位置误差超过逐点30mm（峰32.152mm），不表示全程逐点在容差内。环境与传感器避障门禁仍为false。机器可读状态见 [WBC_READINESS.json](WBC_READINESS.json)。

使用 [OCS2 官方源码](https://github.com/leggedrobotics/ocs2) `ros2` commit `88ce706c17fcc4591b6c24c87a4a85ff6ed9e92d`。固定 URL/逐文件哈希与许可证见 `ocs2/UPSTREAM_MANIFEST.json`、`OCS2_LICENSE.txt`。隔离构建 `thirdparty/core/oc/mpc/qp_solver/ddp` 六包，未升级已有 Isaac Python 环境、未修改上游源码。完整官方移动操作示例的 Pinocchio/HPP-FCL 包未构建；自定义 USD→POE 运动学及真实 collider 保守凸包属于本项目模型，优化器确为原生 OCS2。[官方示例说明](https://leggedrobotics.github.io/ocs2/robotic_examples.html) 用于对照模型范围。

非完整四轮 skid-steer，禁止车体 vy；四个真实轮子与地面接触。预测的底盘仍为平面运动学，不是轮胎/接触动力学模型。 原始关节位置界保持不变。模型的9维位置状态来自每个100Hz PhysX步的实测完整底盘SE(3)及六关节；预测扩展为17/18维，尾部明确定义为**上一条实际下发的速度参考**，输入是参考变化率，不是把理想速度冒充物理测量，也不是完整二阶接触动力学。原始实际 qd/base velocity 另行记录并执行 overspeed、原限位与 qd·dt+qd²/(2a) 刹停余量检查。任何实测限位/速度/余量违例都进入正式总失败，`.015rad` emergency guard 不会把原限位失败改成通过。

底层新增实测 body-yaw PI→真实轮速补偿。原 native yaw 参考上限 .35rad/s、输入率 .65rad/s² 保留；虚拟 actuator yaw 只用于轮速映射，不是放宽 chassis 参考。轮速 8rad/s、轮速 slew=(.45+.65·track/2)/r≈3.85rad/s²、实际车体 yaw .45rad/s 独立审计。记录 native reference / bounded reference / virtual mapping + wheel target / measured 四层；积分在速度/速率饱和与停车时受限。原 friction/contact、32/16 solver 与 maxDepenetrationVelocity=1 保留。

原生 worker 为长期单进程，一次一个在途求解；1s预测时域、ODE初始步长.01s、SLQ最多5次迭代。物理始终继续前进；策略按原始测量绝对时间及墙钟双重.2s lease失效，失败时有界制动、fresh state重算，最多5次失败或2s无新策略后停止并保留失败记录。每次请求只截取原始轨迹中完整覆盖当前horizon及两端插值段的连续knots；没有重采样、移动、重定时或修改相机路径。

公共 `TimedTask` / `LiveTaskSession` 实际控制相机轨迹；world语义相机轴显式转为 optical（+Z向前、+Y向下），保持绝对时间、线性位置/最短SO(3)插值。`accepted_observations.json` 保存接纳时的实际Observation。反馈从实际碰撞GPrim缓存形状、每tick PhysX刚体SE(3)构成移动平台/相机机械臂两部分保守几何；关节、FK、Jacobian留在私有日志。只有 `timed_camera_se3` 已支持；身体区域/构型偏好明确 unsupported，`supported=false, feasible=null`，不会伪装为已证明几何不可行。外部任务仍须绑定新鲜测量Observation；当前内置轨迹是有固定输入、原始任务及哈希的诊断案例。

| 完整原生OCS2物理套件 FULL03 | 结果 |
|---|---:|
| 完整结果 | `results/ocs2_acceptance_03/acceptance.json` |
| 六方向最小沿向位移 | 364.601 mm |
| 六关节独立动作跨度／底盘漂移 | 各约0.239–0.240 rad／4.267 mm |
| 相机位置 RMSE／最大值 | 15.039／32.152 mm |
| 姿态 RMSE／最大值 | 0.008010／0.012986 rad |
| 原Task逐点位置超差帧／姿态超差帧 | 40/2100／0/2100 |
| 最终原Task位置／姿态误差 | 0.592 mm／0.002720 rad |
| 最终停车窗相机位移span | 0.294 mm |
| 相机任务底盘跨度／手臂变化范数 | 0.6702 m／1.6329 rad |
| 物理／墙钟 | 163.16／163.443 s |
| 100Hz实测样本／实际安全违例 | 16316／0 |
| 完整验收、时序、原始RGB及光轴探针 | True |

通过范围依照原始RMS及终点指标，**不是每个轨迹点均满足容差**。逐点超差集中于Task相对18.84–19.23s的40帧（.4s，1.905%），峰32.152mm，姿态无超差；固定5s终点保持后误差为.592mm/.002720rad。`ocs2/evidence/full03_pointwise_scope.json`由原始Task与实测光学SE(3)独立重算。公共反馈保留这些逐点false，没有重写原参考或容差。根级 `validation/wbc_husky_full03_{physical,suite,joint_limits,render}_independent_20261020.json` 分别核验原Task／光学轴／接纳ID与反馈／全部DOF及source哈希／时序、六方向和独立臂及每段停车、原始限位、真实渲染。独立臂各关节与全部方向从raw重算，不依赖runner的passed字段。

严格停车预声明5s预算，检查末.5s全部真实DOF（包含Husky轮子、Ridgeback的m/m/rad底盘）及底盘/相机完整SE(3)：位置span<5mm、两两姿态span<.01rad、转动关节span<.01rad、移动副<2mm、最终实测底盘全线速度<.02m/s和角速度<.03rad/s；pace要求wall≤1.05×sim。不会只检查机械臂、缩短窗口或丢弃settle。独立审计脚本重新从物理日志与原始Task计算结果。

CPU理想模型闭环另存 `results/ocs2_model_home_03/summary.json`：21s完成，位置RMSE 2.384mm、姿态RMSE 0.007005rad、最长solver 0.0881s。这是模型预测结果，与上表物理精度分开。62项CPU后端测试包含真正原生持久solver错误恢复/绝对时间、随机解析导数、凸包守卫、实测qd制动、轮式servo及不可达参考拒绝；初次失败、旧单horizon结果和CPU版文档保留在 `ocs2/evidence/`。

关节/速度/输入率在SLQ内仍为 relaxed barrier **软约束**（mu=.2、delta=.005），不是硬约束。发布前按原始硬上限检查所有策略knots；刹停代价及真实碰撞凸包距离局部线性化参与优化，全部非固定/非相邻collision pairs保留，发布前再做非线性凸包检查，物理每tick另有真实形状guard。凸包是保守外包，仍可能误拒绝；knots及100Hz检查不构成连续域全工作空间保证。未建立力矩、驱动延迟、接触/滑移的完整预测模型。驱动力/速度cap已配置并读回，实测速度单独审计；projected reaction effort不能等同隐式PD实际电机扭矩，因此公共actuation正向状态保持unknown。

环境约束与真实LiDAR避障未验证，传感器已有有限姿态/渲染证据与已知盲区仍保留；environment与sensors_avoidance gates为false，未知不会填pass。历史 `results/acceptance_05/acceptance.json（local constrained differential QP）` 仅证明旧控制器，不替代新OCS2验证。

```bash
# 当前目录内；固定来源隔离构建，不升级现环境
./ocs2/scripts/build_native.sh --mpc
# 新建输出目录；--wbc-only 仅相机诊断，默认含方向/独立臂/相机/每段严格停车
./run_ocs2.sh --wbc-only --output /absolute/new_camera_result
./run_ocs2.sh --output /absolute/new_full_result
```

原生内部单horizon入口 `ocs2/scripts/run_native.sh` 仍可复现旧CPU案例，但不会发送物理命令。新入口冻结执行源、原生binary/guard、模型/传感器配置、完整原始Task及source SHA；流式记录全部DOF、真实速度、body/camera/reference SE(3)、约束反馈和分项wall耗时，结束保存flattened tested scene。`ocs2/.cache`与大体积results被忽略，但构建脚本/固定manifest/持久CPU结果保留。

FULL01的负向记录保存在 `results/ocs2_acceptance_01/` 与 `ocs2/evidence/full01_pure_turn_overspeed.json`，未删除63条重复pre/post审计或31个不同超限tick。新原地turn模式把运行轮速目标限为1rad/s，实测|yaw|≥.22先制动至≤.12再恢复；所有零轮速目标仍经过原slew。原heading角度、60s期限、.015/.04rad验收、直线运动幅度和实际.45rad/s硬限均保留。相机模式与已通过camera03冻结源在1000组相同输入下输出逐位一致；当前单一地面版本另经完整FULL03真实物理重新验收。

终点控制现在显式保留原Task全部knots和绝对end时间：预测域越过end后只恒定延续同一终点。固定5s停车预算的前4s，原生SLQ继续按新鲜实测状态修正末端，底盘按原slew制动；最后1s全部命令按原加速度制动。原生state-input equality约束同时描述已施加的制动；输入rate归零后模型固定当前测量底盘/关节条件。ModeSchedule仅在已知制动完成时刻分段，jump为identity，不重置物理状态。最终Task位置/姿态必须小于已接纳Task自己的容差；RMS、最大值、逐点超差数与终点结果分别保存。

历史FULL02的RMS与方向/停车汇总虽过，独立审计发现终点30.241mm超过30mm，正式验收失败；camera04终点过但停车span5.463mm失败，camera05终点/停车/pace过但早期物理接触脉冲超原速度界，亦失败，全部原始文件保留。真实contact诊断发现SDK GroundPlane同时启用了visual geom与analytic Plane两个共面碰撞面，且Plane未绑定指定物理材料。当前仅保留原analytic Plane并绑定原.8/.6材质，visual mesh只渲染；所有机器人形状/碰撞对/offset/drive/caps不变。`ground_audit.json`在reset前后都强制验证只有1个floor collider及正确材料。Ridgeback同源场景也修复并须重新完整物理验证，不能复用旧分支等价证明；Husky当前FULL03已实际重跑通过。

实际cooking API读回的18个Mesh代理与raw source convex hull的逐面半空间对照通过，最大外出仅浮点1.032e−8m；`results/ocs2_contact_analysis_01/`保留完整局部/世界顶点、变换及源SHA，根级独立mesh-local核验亦通过。因此当前guard/公共外包围盒仍为保守source hull，不能笼统把raw hull称为PhysX精确cooked代理。一次contact诊断未复现camera05脉冲，只能证明双地面确实同时产生过真实轮接触；未复现不是反证。contact报告开销造成29s/42.578s计时失败，诊断不作为新验收通过。
