# Go2-Piper：真实 learned WBC 与 WORLD 相机协调

更新：2026-10-20。实际构型是 **Go2 + Agilex Piper 6DOF**。按用户最后调整，本轮以基础稳定控制与接口适配收尾，不再继续精密停车优化。基础层和原严格门禁分开：当前完整九项基础运动、相机任务接口与实时性已实测；严格精密停车仍为 False，不能作为任意相机/身体控制器或避障 Agent 已就绪交付。证据绑定见 `WBC_READINESS.json`。

## 最终基础配置及范围

`run_basic_wbc.sh --output results/NEW_RUN` 固定已测的完整 92 秒配置，`--preflight` 只检查当前源码闭包，不启动物理。该薄入口保留原 native runner 的严格结果与退出码；基础可用不把原 `all_pass=false` 改成 true。当前 `adaptive_runner.py`、协调器、私有预测/制动/反馈、实际使用的公共协议/几何模块等 23 个控制来源与 `results/adaptive_suite_planar_brake_01/sources` **逐字一致**，映射见 `basic_control_profile.json`。未执行的 G1 backend/Pink/upper 文件仅保留在历史归档，不作为 Go2 已执行后端或 live 依赖。

最终 profile 保留原 18 个 learned actions、200 Hz 物理 / 50 Hz 策略 / 四子步写回，原模型、碰撞对、力矩和速度上限。原生命令上限 `[.2,.2,.2]`，50 Hz slew `[.5,.5,.8]`；停车腿部原 D 倍率为 3，机械臂倍率 1。相机反馈原 P 与 `Ki=.2` 共用全三维 25 mm correction cap。新任务准入用新鲜实际已消费命令做有界输出连续初始化，完整 Task SHA、同刻相机/link/外参、60 ms 实发时效与事务回滚仍必须满足；这不是静态标定或训练稳定域证书。

停车原五秒预算不变：前 1.6 秒从属 XY/yaw 分配，0.4 秒原限幅制动；后 3 秒允许 `vxy_p05_v1` 以同刻实测 body `vx/vy` 施加 `[-.5vx,-.5vy,0]`，每平动轴 assist cap .03 m/s，继续服从原 input caps/slew、原 1.2 秒七点工作域预测、所有 FCL/时效/域恢复 veto 与同 tick ZI 回滚。预测使用实际首命令和 150 ms 输入历史，未来反馈仅作冻结当前反馈目标的运行近似，不是 learned/contact 预测认证。没有冻结 18 个动作，也没有写入底座或关节状态来停车。

| 最终完整实测 | 数值 |
|---|---|
| 原九项速度 / 相机 RMSE | 9/9 / 9/9 通过；最差位置 RMS 23.192 mm、姿态 RMS .020274 rad |
| 原限位 / 驱动 / 自碰撞 / 实际输入外域 | 各 4500/4500 fresh pass；0 infeasible feedback |
| 时序 | 92.0 s 仿真 / 92.000083 s wall，原 5% 实时门槛通过 |
| 原精密停车 | 7/9；左转与 camera_shot 失败，原门槛与 False 保留 |
| 原末半秒最差 span | 机身 4.106 mm / .007860 rad；相机 11.171 mm / .008729 rad；18 关节最大 .025066 rad |
| 各段末速最差 norm | 平动 .015619 m/s；转动 .037018 rad/s |

基础层独立粗停车判据为机身 10 mm/.03 rad、相机 20 mm/.05 rad、关节 .05 rad（若有直线轴为 .01 m）、最终线速度 .03 m/s/角速度 .05 rad/s；保留原相机 RMS/末端容差、限位/驱动/FCL/输入时效与实时门槛。基础方向判据仅要求每个请求非零轴的实际均速同向且幅值至少 .01；它不承诺严格速度精度。原精密门槛仍是 5 mm/10 mrad、关节 10 mrad、末速 .02/.03，**没有改写历史验收**。

根级独立 `wbc_go2_planar_basic01_independent_20261020.json` 重算九项基础通过、`audit_errors=[]`，其内保留 physical 原严格 False、实测全关节位置/速度 limits True 和公共反馈一致性 True；独立 `wbc_go2_planar_input01_independent_20261020.json` 确认原实际 XY 反馈公式、时钟、限幅、slew 与 veto 路由通过。`basic_control.qualified=True` 仅对应这个降低后的范围，所有旧严格门禁保持 False。

## 公共任务、时效和不支持项

当前物理会话仅声明 `timed_camera_se3`。接口接纳后冻结原 WORLD 绝对仿真时间轨迹、观测 ID、路径和容差；相机语义轴 x前/y左/z上在边界显式转光学 x右/y下/z前，终点后明确 hold 原终点，不外推或重定时。每帧返回实际相机/误差及全部真实 collider 组成的 `mobile_platform` / `camera_manipulator` 两部分几何。

`LiveTaskSession` 的观测/约束最大年龄 .1 s；晚于当前时刻或过期证据拒绝/降为 unknown。seed 准入实发证据最大 .06 s；当前控制周期 .02 s，不把上次结果当成新鲜结果。原生外域 x[.4,.7]、y[-.35,.35]、WORLD-z[.1,.8] m 与相对 Euler [-.785,1.04667] rad 只作配置外包络，域内训练覆盖与动力学可行性仍未知。

`supported=False` 表示缺能力/不支持（如任意 body pose、区域约束、接触规划），`feasible=None` 表示尚未评估，不能当机器人几何不可能；原实际请求/预测域违反返回 `out_of_policy_domain`，原约束或求解拒绝对应 `solver_infeasible` / `configuration_limit` / `actuation_limit` / `self_collision`，过期分别为 `observation_stale` / `command_stale`。环境地图与避障传感器充分性没有本轮证据，反馈 `unknown`，`environment_constraints_verified` / `sensors_avoidance_verified` 保持 False。RGB + 光学正 Z 探针只证明该实测时刻渲染方向，不证明全姿态无遮挡、全 LiDAR 覆盖或避障。

本 profile 未追加原 16 秒明显六维轨迹复验；早先其它 profile 的 6D 正例单列为历史，不拼接成当前精密验收。`backend_runtime_ready`、`continuous_camera_tracking_verified`、`representation_ingress_verified` 三项原严格完成门禁继续 False；`basic_control` 单独描述这次基础范围。

下面技术过程与结果表作为保留的历史证据；其中旧 driver SHA 与旧 profile 结论不替代上述最终基础配置。

## 原生后端及来源

`run_adaptive_wbc.sh` 使用 LeggedManip_Lab `GO2-PIPER-WBC`：210输入、18输出（12腿+6Piper），原动作scale=.25、原200Hz物理/50Hz策略。policy SHA256 `c5f57c79bc01d8b8f2c0846163b149e3f53de39291770a1ccb230ef3be6b800e`。没有换flat/D1/ARX5权重，没有用另一个IK覆盖18个learned joint targets。原 `results/validation_03` 80s证据只对应旧混合坐标任务，保留原样。

逐joint力矩来源明确区分：腿部USD/XML caps一致；Piper源USD drive为100Nm，而pinned `mujoco/robots/go2_piper/go2piper.xml` 的unit-gear motor ctrlrange为 **20/20/15/7/5/5Nm**。本轮在explicit actuator model与simulation两层均采用这些XML motor caps，位置范围和权重未改。`declared_motor_limits.json`、`source_usd_drive_caps.json`、`actuator_audit.json` 保存实际值和来源。不能说原USD手臂是20Nm，也不能把旧默认simulation maxForce=1e9直接等同旧explicit model没有clip。

显式 applied torque 是实际下发给PhysX的有界命令，不是硬件力矩传感器。每200Hz substep记录effort与实测qdot峰值，原生force/speed饱和的剩余余量记录为0；full suite确有effort fraction=1.0，不能用位置余量冒充动力余量。原生Go2 self-collisions=False，因此零contact力不被当成碰撞安全证据。

## WORLD相机与混合坐标

原生命令为 `[x_link0,y_link0,z_WORLD,qw_link0,qx,qy,qz]`。先用完整光学相机外参得到末端pose，再换算实测link0混合命令；有roll/pitch时逆转换要联立二维方程，不能旋转xy后覆盖z。真实相机挂在 `end_effector`，平移 `[.075,-.04,.12]m`，完整外参保存在每次冻结rig。物理光学轴x右/y下/z前，公共语义轴x前/y左/z上，边界显式转换。

原始公共 `TimedTask` 是WORLD绝对仿真时间，冻结后不改，相机平移/shortest-SO3插值，末时刻后显式保持原终点。每tick实测全部相关body，反馈 `mobile_platform` / `camera_manipulator` 的真实collider占据体；不是虚拟稳定相机。所有参数、policy、共享协议、raw测量与展开场景冻结，pace与安全/跟踪分开验收。

`CameraCoordinator` 对原世界轨迹进行前瞻，协调可支持的base速度输入，不改相机样本。`PolicyDomain` 使用固定上游配置外包络：x[.4,.7]、y[-.35,.35]、world-z[.1,.8]m，原sampler最终相对Euler clamp[-.785,1.04667]rad。**域内仅未被外界排除，checkpoint实际训练覆盖和动力学可行性未知。**

真实倒退试验表明初始response约.5–.6，过去把input当实际速度会耗尽arm余量。`measured_backward_v1` 采用负vx前馈1/.55，保持native input≤.2及slew(.5,.5,.8)；预测使用实际输入历史、gain .55、tau .35s、delay .15s和未来受slew约束的输入ramp。这是局部运行近似，不是训练域或动力学证书。

被拒绝的P校正、原WORLD换算request、**实际送入policy的7维命令**和对应domain分别保存。域外请求不再送入policy：明确保持最近合法native姿态、feasible=False，在2s预先固定预算内用新鲜状态计算有限base恢复输入；失败后制动，不无限复用旧速度，不改原相机轨迹。不可恢复的高度/roll/pitch不会编造身体控制。任何已知constraint fail或feasible=False均使整体False。

## 实测证据

| 目录 | 结果 |
|---|---|
| `results/wbc_camera_coordinated_01` | 原固定WORLD小相机+持续移动失败：camera .33064m/.06077rad，200个active命令156个域外。原任务与失败保留，不能改成共移目标后说修好了该不兼容任务 |
| `results/adaptive_forward_04` | 第一个forward+strictstop+pace通过，camera18.53mm/.01606rad；未转移为全方向或环境认证 |
| `results/adaptive_suite_fine_01` | 完整92s/9case：所有速度与camera RMSE通过，4500/4500 active FCL fresh通过、pace=.99325；strictstop仅forward通过，其余8项失败。10帧旧domain反馈误报来自被拒绝P校正，后续源码已分开记录实际输入和拒绝候选，旧日志不补写 |
| `results/adaptive_parking_01` | EE输入锁存+anchor对照未修复跨段停车，负例保留；锁存恢复为非默认可选 |
| `results/adaptive_backward_pd_01` | 原预测的单独倒退响应不足，3.78s开始原request域外，camera136mm；正确False，不归因于只在stop阶段改变的PD |
| `results/adaptive_backward_response_01` | 新实测响应模型，完整12s：尾均vx=−.099878（目标−.1），camera12.86mm/.01353rad，500/500 configuration/actuation/FCL/实际domain通过，0 task infeasible，pace=.98917；strictstop仍因RL hip .01898rad、FR thigh .0163rad、joint2 .01243rad及最终角速.06894rad/s失败 |
| `results/adaptive_backward_anchor_01` | 相同响应模型+新park anchor，12s完整、相机/速度/FCL通过；strictstop仍因base旋转span.01499rad、RL hip .01426rad、最终角速.07947rad/s失败。原5s停车负例保留 |
| `results/adaptive_backward_velocity_audit_01` | 只读200Hz诊断，2400 substeps。原始root与base均为`/World/Robot/base` index0，pose/twist、缓存/API与FK前后差均为0；q轨迹与anchor01完全一致。真实角速在相邻5ms步交替，形成100Hz余振，50Hz只采其中一个相位。不是错刚体或时间戳，不能用均值/pose差分替换原角速门槛。探针额外开销使pace=.884，正确未通过 |
| `results/adaptive_backward_pd_native_arm_01` | 同响应/anchor，停车legs3×/arm1×去掉100Hz交替，速度/camera/500fresh FCL过；原5s停车仍因baseRspan.01954rad、qmax.02121、finalω.03851失败。末窗探针pace=.95214略未过 |
| `results/adaptive_backward_lowpd_hold8_01` | 事前固定8s停车、同低PD+EE去抖：650fresh FCL/移动相机/速度过、0 infeasible；停车仍失败（base5.65mm/.0141rad、camera12.49mm、qmax.03077、finalv.03115/ω.10391），pace=.9435。8s与去抖未构成有效修复，仍非默认 |
| `results/adaptive_backward_force_01` | 单变量force-every-iteration、原4/0：速度/camera/500 FCL过，但停车qmax.04930、finalω.10181失败。官方建议不是自动修复 |
| `results/adaptive_backward_ground_01` | 单变量ground combine匹配multiply，保持系数、forceFalse、原4/0：速度−.10075/camera12.41mm、500 FCL过，停车qmax.03056/finalω.04482失败；末窗额外探针使pace=.9251，保留False |
| `results/adaptive_backward_precision_01` | ground multiply上独立forceTrue+scene velocity floor1复合精度候选；实际stage读回4/1有效floor。速度−.09998/camera13.19mm、500 FCL过，但停车qmax.04372/finalω.07496、pace=.9420失败。未开启额外200Hz探针，未提升默认 |
| `results/adaptive_backward_softpose_01` | 原ground/native4:0/forceFalse/低PD3:1、仅增加移动期x=.58从属偏好：速度−.10655、camera12.90mm/.01142rad；18DOF严格停车首次通过（qmax.005774，base1.538mm/.001189rad，finalv.00205/ω.00258），500 FCL/实际domain全过、0 infeasible、RGB通过。pace=.9460仍失败，整体False，不能转移为全方向通过 |
| `results/adaptive_backward_softpose_02` | 相同控制/物理/原Task，只改等价的pose-only采样；实测速度/相机/全部18q停车与softpose01逐值一致，控制均耗7.043→4.332ms，pace=.999993，500 fresh FCL/实际domain/RGB通过。该12s倒退局部验收整体True；完整九项与大6D仍须同版验收 |
| `results/adaptive_suite_softpose_01` | 完整92s/九项：所有速度、camera与4500 fresh FCL/实际domain/pace=.99874通过，0 infeasible；forward/backward/left/两diagonal停车通过。right/yawL/yawR/camera_shot qspan .01435/.04048/.03884/.02445rad、最终角速均超原门槛，整体False |
| `results/adaptive_camera6d_softpose_01` | 同driver/控制，23s明显6D：camera29.65mm/.01596rad，实测span.17244m/.46970rad，hold误差比.4356/.05788、arm qspan norm1.60345rad，pace≈1/1050 fresh FCL。moving base32.26mm/.31771rad平衡漂移；末停车base7.32mm、camera15.93mm、qmax.03495/finalω.03873失败，且131个move帧future-support solver_infeasible，实际native域合法也不能掩盖不可行反馈。整体False |
| `results/adaptive_camera6d_prediction_01` | 修正持续future-input预测并保留原节点扩展至1.2s，x58偏好也用于零preferred-base相机任务；原23s参考保持。camera21.34mm/.01730rad，实测span.18849m/.47637rad，hold误差比.31354/.06274，arm qspan norm1.55006；1050 fresh FCL/0infeasible/pace≈1，全部18q严格停车max.000947rad通过。moving base103.81mm/.21421rad，为真实全身协调，不能称固定底盘独立臂运动 |
| `results/adaptive_suite_prediction_01` | 同预测profile完整92s九项：全部速度/camera/4500 fresh FCL/实际domain/pace≈1，0infeasible；前五方向停车通过。diagonal_right/yawL/yawR/camera_shot qmax.02415/.03376/.03535/.05672、finalω超原门槛，整体False；单项6D通过不能转移为全套完成 |

原始27形状FCL按实测body T检查所有源非过滤pair，只接受源filtered/disabled-adjacent排除，不按home采样删pair。离散fresh检查与前瞻domain不是连续swept碰撞或环境地图认证。环境反馈保持unknown。

停车原最后完整.5s门槛不变：base/camera平移span<5mm、完整SO3 span<.01rad、全部18DOF span<.01rad、最终线速norm<.02m/s/角速norm<.03rad/s。停车native PD damping可平滑调整，所有learned targets、原caps保留，输入模式/每组倍率写入raw。

200Hz角速末.5s奇/偶substep均值分别约`[-.063,-.0175,.0487]`与`[.0777,-.0055,.0049]rad/s`，瞬时norm约.08，均超过原.03门槛。证据 `velocity_sampling_alias_audit.json` 明确保留各相位和原生判据；当前优先检查高倍率explicit PD的离散稳定性，尚未把增阻尼或输入锁存称为有效修复。

低PD去掉高频交替；force-only和force+velocity1的真实对照仍有低频余动，均未通过停车。实际floor仅一个Cuboid collider。原机器人材料是实际场景default .5、地面1，average合并有效摩擦.75；`--ground-material source_multiply`只将ground combine匹配固定上游multiply，有效摩擦.5，原系数不变。它没有重建训练时的机器人随机摩擦分布，单独材料对照也未修复停车。

`--go2-native-posture x58_moving_v1`是相机主任务下的从属分配偏好：在原可支持速度输入和原slew内，让预测native x渐近局部实测有效的.58m，进入任务末端后按事前固定.5s渐撤，停车不持续挪base追求该偏好。原WORLD相机样本和base速度请求不改，硬x工作域仍[.4,.7]，原18个learned actions不动；这不是训练/稳定域证书。域外或不可兼容任务仍明确拒绝，原固定WORLD长程前进失败保留。

`x58_all_v1`将同一从属分配应用于零preferred-base的相机任务；`persistent_bias_12_v1`保持candidate与preferred的偏置在预测未来输入中持续有效，保留原0/.2/.4/.6/.8s节点，增加1/1.2s节点。原35mm运行reserve仍在.8s达到，不因更长horizon变小；全部候选仍需真实非线性domain重检、原input cap和slew。新版23s相机任务确实移除了原131帧不可行，但九项连续试验后四段停车仍失败，正式gate保持False。

`optical_sampler.py`仅缓存已经接纳的原TimedTask语义轨迹，按原绝对时钟、线性位置和同一Slerp后转换光学轴，禁止隐式外推。只替换协调器中被丢弃twist的pose回调；公共任务接纳、测量和反馈保留。17项插值/π弧/大绝对时钟/过期/不可变回归通过；`results/optical_sampling_cpu_01.json`在softpose01的100个真实body快照重放中base command差≤1.06e−13、status相同，CPU中位5.056→3.019ms。CPU等价不代表物理实时通过，须新目录重测。正式readiness仍False，旧可运行/公共ingress证据单列previous-profile。

16s `camera_6d` 原参考位置span=.17167m、SO3 span=.46641rad，初始hold RMSE=.06807m/.27577rad。实测响应和原误差门槛通过，但131个前瞻不可行反馈与严格停车失败，不能认证完整相机任务。零preferred-base期间也有32.26mm/.31771rad真实平衡漂移，没有硬冻结机身或用虚拟相机冒充独立臂运动。

`adaptive_runner.py`已从实际FULL冻结driver逐字复制到本Go2目录，SHA保持`85bfd95d4c629ea590aaf89cfaa04135cf47ce2ac84e0df8246fc31267404d2c`，wrapper使用此独立入口。PACKAGE同层路径、self-respawn、冻结依赖静态等价见`results/driver_relocation_01.json`；随后23s6D确实由新入口执行。G1后续优化不会改变这个Go2 driver，旧冻结证据保留。

原接口只支持base速度与末端混合pose；不提供任意身体高度/roll/pitch或通用接触规划。继承rig为3个LiDAR，本轮未改数量/安装；局部RGB/ray证据不代表完整全姿态覆盖或避障充分感知。`environment_constraints_verified` / `sensors_avoidance_verified` 保持False。

固定官方来源：[项目](https://github.com/zzzJie-Robot/LeggedManip_Lab/blob/master/PROJECT_INTRO.md)、[WBC配置](https://github.com/zzzJie-Robot/LeggedManip_Lab/blob/4e12109d1eccb1d9da2e4a35f189fabea3f13677/source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/config/go2_piper/wbc_env_cfg.py)、[混合pose sampler](https://github.com/zzzJie-Robot/LeggedManip_Lab/blob/4e12109d1eccb1d9da2e4a35f189fabea3f13677/source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/pose_command_wbc.py)。本地commit `4e12109d1eccb1d9da2e4a35f189fabea3f13677`。
