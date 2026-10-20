# G1：真实 AGILE height + Pink 上半身控制

更新：2026-10-20。按本轮调整后的范围，**基础稳定控制与接口适配已完成**；最终九项92秒实测与独立基础审计通过。精密速度、全部29关节严格停车及任意相机轨迹的完整发布条件仍未全部通过，原严格结果与正式门槛保持不变。状态和源码绑定见`WBC_READINESS.json`，全部历史失败保留。

## 本轮交付的基础范围

推荐`./run_basic_wbc.sh --output results/my_basic_run`。该入口明确固定已测fresh-current-state重力支撑、moving-yaw Ki1/FF2、原关节范围与控制频率；未运行的Ki4只保留为开发实验，不作默认。本轮最终结果为`results/basic_full_validation_01`，92.000093秒墙钟完成92秒仿真，4500个活跃测量全部具有新鲜实际形状FCL证据；相机原RMS和终点容差、原配置/速度限制及基础方向运动均通过。

| 工况 | 目标vx/vy/wz | 实测后半程均值vx/vy/wz | 基础方向、相机及粗停车 |
|---|---|---|---|
| 前进 | .10 / 0 / 0 | .1053 / −.0021 / .0059 | 通过 |
| 后退 | −.10 / 0 / 0 | −.0958 / 0 / .0011 | 通过 |
| 左移 | 0 / .10 / 0 | .0017 / .1006 / −.0024 | 通过 |
| 右移 | 0 / −.10 / 0 | .0041 / −.1073 / −.0003 | 通过 |
| 左斜向 | .10 / .10 / 0 | .1009 / .0977 / −.0051 | 通过 |
| 右斜向 | .10 / −.10 / 0 | .1033 / −.1009 / .0056 | 通过 |
| 左转 | 0 / 0 / .12 | .0020 / .0062 / .1520 | 通过；精密速度未过 |
| 右转 | 0 / 0 / −.12 | −.0048 / −.0095 / −.1316 | 通过 |
| 移动摄影 | .10 / 0 / 0 | .1040 / −.0024 / .0029 | 通过 |

独立基础审计为`../validation/wbc_g1_basic_full01_independent_20261020.json`，`basic_control_pass=true`、0审计错误。基础停车采用预声明的末0.5秒范围：机身10mm/.03rad、相机20mm/.05rad、全部关节.05rad、最终线速.03m/s及角速.05rad/s。本次实际最大值分别仅2.55mm/3.08mrad、.756mm/2.38mrad、20.25mrad、.01752m/s和.02216rad/s。

原精密速度8/9通过，左转.1520rad/s仍高于.12目标；原严格停车只4/9通过，倒退、双斜向、左转和移动摄影仍因关节span未过原10mrad门槛。`summary.json`的`all_pass=false`及退出码2完整保留；4次真实短暂QP失败也保留，基础审计没有认证其180ms恢复墙钟。它们不再触发本轮后续调参。

公共WORLD/绝对时间`TimedTask`实际进入物理循环，实测光学相机与两个几何模块的`Observation`、限位/碰撞/unknown状态实际返回。最终RGB与光学+Z渲染检查通过，只证明该终点实际渲染；**不证明全姿态无遮挡、环境覆盖或避障**。旧16秒明显6D轨迹属于另一冻结控制版本，不转移为当前基础配置的任意6D能力。环境、全机执行力矩认证和避障条件继续保持unknown/未验证。

## 后端、控制权与原限位

`run_adaptive_wbc.sh` 调用 `adaptive_runner.py`，使用固定 WBC-AGILE v1.2 的 velocity-height recurrent student：80 输入、12 腿输出、命令 `[vx,vy,wz,height]`。原 `run.sh` 的 255 输入/14 输出 velocity-history 策略及 `results/acceptance_01` 证据保留，不能替代新 height 验收。物理 200 Hz，原生策略/控制与测量 50 Hz，pace=1。

真正 `pin-pink 3.1.0` / Pinocchio 2.7 / quadprog 在隔离 worker 内执行。Pink 优化 waist yaw 与左右各 7 个手臂，共 15 DOF；12 腿只由 AGILE 控制，waist roll/pitch 保持。每次锁定实测腿和腰部两个保持关节，pelvis 完整 SE3 仅作测量变换，**不执行自由基座 IK，不修改 learned leg actions**。完整相机外参和公共语义轴转换在边界处理。

本轮使用 height 策略配套 `G1_29DOF` / `Physics=PhysX,left_hand=None,right_hand=None`。其真实运行资产有 **35 个 enabled collision GPrim**；另一默认 variant 的 31 形状静态审计不能转移至此。全部 29 关节位置范围与官方 URDF 一致，未收缩；height 配套 hip-roll/ankle 速度 cap 为 20/10 rad/s，不混用旧配置的 32/37。

力矩来源分别记录：height 源 USD 的 hip-roll 为 139 Nm，本轮采用历史 **88 Nm 保守 operating cap**；不能把源 USD 描述为 88 Nm。其余逐关节值见各结果目录 `source_usd_drive_caps.json`、`actuator_audit.json`。腿仍为原生 implicit drive，实际力矩不可由驱动估计认证。上半身使用一个 explicit PD+模型重力前馈之和，**总和只在原 operating cap 内裁剪一次**，没有在 implicit drive 上叠加额外无界力矩。模型重力是前馈，不是力矩传感器。

## 公共任务与执行

公共相机为 world、绝对仿真时间、x前/y左/z上；真实物理相机是光学轴 x右/y下/z前，挂在 `right_wrist_yaw_link`，完整安装外参来自冻结 rig。没有虚拟稳定相机。原 `TimedTask` 样本冻结，平移线性/旋转 shortest-SO3 插值，超出末时刻显式保持原终点。

每帧基于实测各刚体，生成 `mobile_platform` / `camera_manipulator` 实际碰撞体的保守几何反馈。环境约束没有测量证据，保持 unknown。配置余量与电机/速度余量分开；implicit 力矩仍 unknown。请求、权重、源码、rig、共享协议与结果展开场景分别保留。过期 worker 响应丢弃，不冻结 PhysX 等待，不无限复用旧速度。

`bounded_integral` 候选只积分 fresh Pink qdot，并用实测 q 的 ±.05 rad lead antiwindup 和原物理位置范围约束 qref；零 qdot 时 qref 固定。上半身保持完整 Pink 控制权。进入站立时可一次锁存新的 policy 停车 anchor，避免将正常制动距离误当成持续漂移而反复走步；随后真正 25 mm/50 mrad 漂移仍退出站立。原 WORLD 相机终点不改。

`--park-posture-mode measured` 在实测进入站立时一次锁存 15 上身 q，作为原生 **软 PostureTask** 偏好；原 FrameTask 和全部碰撞 barrier 继续工作，可以偏离这个偏好，不是锁定手臂。该方案的前进与大相机任务已经物理通过，九项测试仍失败。

`--parking-profile brake_settle_v1` 改为停车开始即锁存软 Posture，依次执行预声明的 BRAKE→按原slew降至literal-zero的SETTLE→持续实测稳定才标记PARK；随后真正漂移仍退出。它消除了原滤波速度过零造成的反复PI走步，但尚未通过全部关节停车。固定上游height配置的速度外包络为vx/vy±.5、yaw±1.0；源采样把命令norm<.1置零，因此持续小幅PI走步没有被源配置认证。详见 `results/height_policy_command_source_audit_01.json`。

`--waist-posture-cost .15` 使用真正Pink逐关节软PostureTask，让waist yaw尽量靠walking home0，其他上身cost=.03；不锁腰、不收缩原范围，FrameTask与真实碰撞barrier继续决定全部15关节。CPU对照在同相机目标下将waist末误差从.09195降至.00148rad，同时保留约2.82mm/.0114rad相机误差。新的四项物理对照把camera_shot腰偏转降至.0783rad并修复该段速度，但yaw速度仍未通过，不能称整套修复。

## 实际自碰撞约束

`legged_collision.py` 从实际 USD Mesh 点/primitive、最近刚体、instance proxies、guide-purpose 和最近 CollisionAPI 抽取几何。仅排除原源文件的 filteredPairs / collision-disabled adjacent joints；不按 home pose 删除 pair。35 形状有 **548 个源非过滤 pair**，独立 Python-FCL 对所有 pair 以实测 body T 检查，5 mm 间距为原物理门槛，异步结果有 60 ms lease。

实际形状通过初始化一次测得的固定 body-frame 外参挂到 Pin 模型：7392 个历史测量 body 样本 FK 最大位置差 **0.271 mm**，角差 5.37e-7 rad；21 个时刻的 Pin/HPP-FCL 与独立 Python-FCL 最小距离差最大 **4.31 µm**。真正 `pink.barriers.SelfCollisionBarrier`每次计算473个upper-controllable pair的距离，按原生算法选择最近20对构造速度约束；6.5mm floor保持，早期gain .1/s、本轮基础配置为.4/s。外部FCL仍检查全部548对，包括腿和固定链。该检查是离散实测/速度 barrier，不是连续 swept collision 或地形接触认证。详见 `results/pink_collision_cpu_02.json`。

## 本轮原始物理证据

| 目录 | 结果 |
|---|---|
| `results/wbc_pink_forward_01` | 旧新后端早期 6 s 联合小任务：相机 3.85 mm/.00590 rad；下发 vx=.05，活跃段实际前进仅 3.8 cm，不能说实际 .05 m/s |
| `results/adaptive_forward_05` | 无 fine FCL 的基准：尾段 vx=.11490，camera22.76mm/.11205rad，29DOF停车和pace通过；不能转移为新碰撞安全通过 |
| `results/adaptive_forward_low_gain_01` | 低增益无改进；7.26 s 时 torso↔right shoulder_yaw 实际间距2.609mm触发中止，失败保留 |
| `results/adaptive_forward_barrier_01` | 新35形状/原生barrier，500/500 active fresh FCL通过、camera19.40mm/.08631rad、速度/pace通过；停车仍失败 |
| `results/adaptive_forward_integral_01` | 有界qref+驻车anchor：camera22.59mm/.05892rad，尾段vx=.08633（目标.1），500 FCL全过，pace≈1；base/camera span及最终速度通过，但right shoulder_roll/wrist_roll/wrist_yaw span=.011884/.010099/.011093rad，超过原.01，因此整体False |
| `results/adaptive_forward_posture_01` | 软Posture前进12s全过：尾均vx=.09125，camera24.71mm/.06389rad，全部29DOF最大停车span=.007332rad，500 FCL全过，pace≈1 |
| `results/adaptive_suite_posture_01` | 完整92s/9项，4500/4500 fresh FCL、配置和policy外界检查通过，0 infeasible，pace≈1，RGB/probe通过；仅forward全部通过。left停车反复入/退出，尾base/camera span46.8/85.4mm；yaw速度约±.08低于原±.12；其余多项关节停车超.01rad。整体False |
| `results/adaptive_camera6d_posture_01` | 完整23s（16s任务+5s停车+2s settle）全过：camera6.617mm/.032307rad；参考span.171669m/.466408rad，实际.171695m/.469907rad；hold基线误差比例.0972/.1172，arm qspan norm1.580rad；1050 fresh FCL全过，29DOF停车最大.008007rad，pace≈1，RGB/probe通过 |
| `results/adaptive_left_settle_01` | 新停车单独left12s：vy=.08423、camera33.73mm/.08372rad，500 fresh FCL和pace过；225tick持续SETTLE literal0、0错误PARK，但base旋转span.01766rad和qmax.02851rad停车失败 |
| `results/adaptive_waist_settle_01` | 4项42s：2000 fresh FCL、camera全过、0 infeasible、pace≈1；left仅qmax.03099失败，camera_shot vx=.10432速度过但qmax.01586失败；yaw_left停车过但wz=.04655<.12，yaw_right wz−.02388未过且停车超限。整体False |
| `results/adaptive_left_force_01` | 只改变TGS每迭代力积分，原8/4保留：vy=.09641、camera29.15mm/.07203rad、500 fresh FCL、pace≈1；qspan.02723rad、base平移5.93mm，严格停车仍失败，不提升默认 |
| `results/adaptive_recovery_sameprofile_01` | 新恢复通信首次物理对照在6.12s提前退出，worker deadline/重复失败；四项未完成。最后logged seq仍fresh，原证据不能区分下一次主线程wall停顿与worker响应延迟，不声称已验证停车 |
| `results/adaptive_recovery_sameprofile_02` | 修正有界恢复，完整42s、2000 fresh FCL、0 infeasible、pace≈1、RGB通过；left vy=.10214、camera_shot vx=.10643，相机四项全过。yaw仍仅+.03611/−.03176rad/s；四段全部关节停车span .01990/.01637/.01736/.01902rad均超原.01，left最终角速.03785也超原.03。因此仍False |
| `results/adaptive_ground_sameprofile_01` | recovery02上只匹配ground multiply：42s、2000 fresh FCL/pace/RGB通过；left/camera_shot速度通过，yaw改善为+.07846/−.07052仍未过原±.12。yaw_left全29DOF停车通过qmax.003526；其余三段qmax.02206/.02697/.01269及left baseRspan.01344失败，整体False |
| `results/adaptive_pink_soft_profile_01` | walking Posture .1 + locomotion-stop Damping .1组合，42s/2000 fresh FCL/pace/RGB过；left和移动摄影速度过，但yaw仅+.06025/−.05513，四段qspan均仍>.01rad，整体False，不提升walking .1默认 |
| `results/adaptive_yaw_gain2_01` | 仅原生yaw输入前馈增益2，原cap/slew/request保持；22s/1000 fresh FCL/pace/RGB过，yaw+.05606/−.06733仍失败。yaw_right停车过，yaw_left未过；未提升该增益默认 |
| `results/adaptive_actual_gravity_01` | 首次新gravity启动在计时前因Kit/Pin二进制依赖冲突退出，没有物理样本；失败源和错误保留，验证已移隔离worker |
| `results/adaptive_actual_gravity_02` | 仅重力补偿质量/COM匹配真实native，42s/2000 fresh FCL/pace/RGB过。四项速度首次全过，yaw+.09638/−.09106且两yaw全部29q严格停车通过。left qmax.017388/最终ω.04142、移动摄影右肩roll qspan.021444仍停车失败，整体False |
| `results/adaptive_actual_gravity_terminal_01` | 新Task-context绑定/真实状态反馈与locomotion-stop Damping .1组合：42s/2000 FCL/pace/RGB过，两yaw停车过但yaw速度未过；left/移动摄影停车未过，另有5次真正Pink QP无解。公共solver_failure如实计入失败，不以小相机误差掩盖 |
| `results/adaptive_actual_gravity_qpdiag_01` | 同控制前两case22s，仅增加失败持久化：1000 FCL/pace/RGB过，left/摄影停车仍失败；14.24s原请求产生一次QP无解，原始110条不等式/P/q、实测state、previous、Task及pair已捕获，独立LP也判原约束不相容。没有清previous/删除barrier/弱化加速度或重试执行 |
| `results/adaptive_actual_gravity_gain_terminal_yaw_01` | class-K gain .5/s +terminal Damping .2 +原生yaw FF2组合，42s/2000 FCL/pace/RGB过；yaw_left速度+.11710过、yaw_right−.08626未过，四段严格停车均未过，另有一次request23.24s的原QP无解。6.5mm/5mm距离、原q/v/a/effort和所有源pair保留；组合未提升正式默认或gate |
| `results/adaptive_actual_gravity_phase_lease_01` | gamma1 +stop Damping .5 +移动期yaw FF2/3、停车FF1及完整20ms/4子步双lease守卫：27.18s因torso↔right shoulder yaw实测间距4.991mm触发原5mm门槛中止，四项未完成，整体False。left/摄影速度和相机过，关节停车仍分别14.78/11.03mrad失败；两次真QP无解另有原Problem保存。中止前1259个active采样FCL通过不能抵消末次异步fail，原`constraint_abort.json`完整保留 |
| `results/adaptive_actual_gravity_upper_damping5_01` | 上一profile仅上半身执行阻尼3→5：42s完整、2000 FCL/pace/RGB过、最小离散间距5.972mm，没有触发5mm中止；left/摄影速度及四项相机过，但双yaw+.16602/−.15819过驱，四段29q/baseR严格停车均失败，另有3次真QP无解。不能把这次未中止当成已保证全身安全 |
| `results/adaptive_actual_gravity_lower_damping1_01` | D5上仅腿部停车阻尼3→原生1：27.02s因实测4.99905mm触发原5mm中止，summary明确绑定abort及public self_collision/False。left速度/相机过但qmax24.976mrad停车失败；摄影速度/相机及29q严格停车通过(qmax9.539mrad)。yaw_left完成运动但停车尚未完成便中止，完整四段仍False，未提升D1默认 |
| `results/adaptive_actual_gravity_following_ki2_01` | 新条件式following reserve与温和moving-yaw Ki2候选在2.62s因连续3次真正Pink QP无解退出，未完成第一段；没有触发5mm物理间距中止。原始130行Problem与terminal未采样请求分别保存，不能用短前缀的FCL通过称整段通过 |
| `results/adaptive_actual_gravity_arm_stiffness2_01` | 关闭未验证的附加reserve，原CBF/FCL保留；14臂Kp×2、D5、moving-yaw Ki2/FF1。实际PhysX armature=.02999999933、原5/25Nm与速度cap、engine PD0启动读回通过。42s/2000 fresh FCL/pace/RGB及四项相机通过，最小离散间距5.959mm；left/摄影/右yaw速度通过，左yaw+.16998过驱，2次真QP无解。四段严格停车qmax26.94/10.75/11.55/12.84mrad均未过，整体False |
| `results/adaptive_best_gravity_stiffness2_01` | 回到原gravity02 D3/gamma.1/默认terminalD.02/Ki1FF1，保留新Task/lease正确性，仅14臂Kp×2：完整42s/2000FCL/pace/RGB及四项相机过、0QP无解。left/摄影速度过，双yaw+.06954/−.07062未过；yaw_left全部29q停车过(qmax4.67mrad)，left/摄影/yaw_right分别12.00/24.33/16.43mrad失败，整体False。没有将历史profile的双yaw通过转移到本次 |
| `results/adaptive_left_stand_yaw_assist_01` | best+Kp2上只加入PARK偏航反馈，原left12s完整通过：vy=.09873、camera22.06mm/.06649rad、500FCL/最小7.250mm/0QP、pace/RGB通过；29q停车最大5.370mrad，base旋转1.274mrad，最终ω=.003911rad/s。原25mm/50mrad退出保留，实际输入仅−.02289..0rad/s；不能把单左移通过转移为完整方向或新版6D通过 |
| `results/adaptive_stand_yaw_native_stiffness_01` | 上一PARK方案回原臂Kp1，原四项42s完整：2000FCL/pace/四项相机过，1次真QP无解。left/摄影速度过，双yaw+.05914/−.06163未过。四段base旋转压到1.23/1.15/1.37/2.21mrad且最终角速均过，但上肢为主的qmax13.94/21.01/19.55/10.42mrad仍停车失败，未提升默认 |
| `results/adaptive_stand_yaw_symmetric2_01` | 新moving-yaw FF2配置在物理计时前因入口factory映射遗漏退出；没有测量样本/summary，不算物理对照。源与error保留，配置选择已修正并由真实runner factory/VelocityServo正负移动与停车恢复回归覆盖，下一独立目录待实际运行 |
| `results/adaptive_stand_yaw_symmetric2_02` | 修正入口后原42s完整运行：Kp2/D3/gamma.1/默认D.02/PARKassist，只moving非零yaw用对称FF2，Ki1和原cap/slew保持。2000FCL/pace/RGB及相机四项过，1次真QP无解。left速度.09893及29q停车3.30mrad全过；摄影速度.10541过但q22.76mrad；yaw_left+.09245速度过但q13.62mrad，yaw_right−.08344速度与q21.16mrad仍失败。四段base旋转均1.35–1.76mrad且最终角速过，残余上肢停车未完成，整体False |
| `results/adaptive_stand_yaw_park_posture_01` | 上版只在每Task首次原PARK接纳时刷新一次同stamp实际15q软偏好：42s完整，四项速度(.09933/.10669/+.10629/−.09147)、相机、2000FCL/pace/RGB过。left/yaw_left完整停车过(q7.77/5.13mrad)，摄影/yaw_right仍因q12.69/13.80mrad失败，另有5次真QP无解。所有base/camera姿态及最终原速度门槛过，仍整体False；没有把改善当成整套通过 |
| `results/adaptive_park_posture_damping03_01` | 保留首次PARK刷新，仅停车DampingTask .02→.03：42s完整，四项速度(.09933/.10724/+.09125/−.09010)、相机、2000FCL/pace/RGB过，4次真QP无解。left/yaw_right停车过(q7.90/9.57mrad)，摄影q12.00mrad、yaw_left q12.10mrad仍失败；左转末窗一次非fresh上身hold伴随相机span5.72mm/15.81mrad，不能只归于软任务慢收敛。该hold仍按原接口v0/g0执行，独立重算当前状态重力支持的修复尚未接入，整体False |
| `results/adaptive_park_posture_damping03_gamma04_01` | 只将原生CBF gamma .1→.4/s：42s/42.00008wall、四项速度(.10006/.10583/+.11323/−.09275)、相机、2000FCL/RGB通过，最小离散间距6.320mm；1次真QP无解(request33.04s)，仍如实记录。left/摄影/yaw_right全部严格停车通过，29q最大3.95/4.90/3.50mrad；yaw_left全部29q最大8.84mrad及base/最终v/ω过，只相机SO3span10.4215mrad略超10mrad。其末窗31.66s缺fresh并g0时双肘跳4.68/4.09mrad，独立静态重力补偿尚未接入，本轮严格整体False |
| `results/adaptive_fresh_static_gravity_01` | 只增加每5ms当前SDK重力支撑：原四项42s/2000FCL/相机/pace完整通过，四项严格停车全部通过（最大29q span8.31mrad）；实际yaw+.06867/−.06071未过精密速度，2次真实QP失败保留。8400次physical write独立合同全部通过，其中75次local静态力矩与独立Pin1125项误差≤1.45e−15Nm；不把静态模型吻合转移为速度通过 |
| `results/basic_full_validation_01` | 最终同freshg/Ki1配置仅扩九项92s；基础独立审计通过，4500freshFCL、全部原相机/方向性/粗停车/pace通过。原严格速度8/9、严格停车4/9，4次真实QP失败，原all_pass=false保持。此后仅文档与绑定收尾，不再调参 |

停车固定使用最后完整 .5 s：base/camera平移span<5mm、完整SO3 span<.01rad、**全部29DOF** span<.01rad，最终线速norm<.02m/s、角速norm<.03rad/s。不挑选有利窗口，不把临界失败写成通过。新试验可事前另固定更长预算，旧失败不改。

`camera_6d` 的preferred base速度为零，moving段实测base位置span31.71mm、旋转span.12112rad；相机相对base位置/旋转span.19831m/.51541rad。因此确有明显上半身独立运动与跟踪，仍包含真实平衡漂移，不声称硬固定底盘。原50mm/.15rad误差门槛和全部关节停车标准没有改变。

该大相机通过属于旧`parking-profile legacy / waist cost None`精确profile；新版候选未沿用它作正式通过证据。`WBC_READINESS.json`正式版本gate暂False，精确参数和冻结源保留在`previous_profile_validation`，待同版完整验收与默认入口统一后再提升。

通信恢复使用原60ms双时钟lease、180ms无返回边界，最多一个取消旧请求和一个新鲜请求。首次缺帧的旧seq永远不执行；停车后新观测请求可在原lease内等待。先排空已readable完整响应，再区分过期丢弃与无响应卡死；不等待不完整JSON。缺帧时上身qref保持、qdot目标置零，底盘按原slew制动，PI冻结而非瞬时归零并大幅unwind。真实子进程双包/半包、错seq、双钟超时、主线程停顿后fresh恢复等回归通过。`results/yaw_response_diagnostic_02.json`用完整原5s区间重放旧PI，逐fresh输入误差0；但跨方向延迟模型不能可靠辨识，因此没有据单个尾均值添加yaw前馈。

实际floor为一个Cuboid碰撞体，没有双重共面Plane。当前机器人材料实读为场景default .5、地面1，旧average有效摩擦.75；固定上游ground combine为multiply，对同系数有效摩擦.5。两机器人均已单独测试此源匹配；G1转向有所改善仍未通过。这只匹配ground combine，不能声称重建训练随机摩擦分布。力积分候选未修复停车，仍保持native forceFalse。

原官方URDF的35.115142kg质量模型包含当前height USD没有安装的两只rubber hand及不同torso质量。真实PhysX初始化测量30body总质量33.001145kg。`gravity_model.py`仅构建独立gravity-only point-mass模型：先清除原融合惯量，再将全部真实body质量/actor-local COM按标定外参聚合；不改变USD质量、惯量、IK/FK拓扑、碰撞模型或12个learned actions。启动必须核对USD哈希和当前30body实测质量、COM及FK，任何不匹配拒绝执行；核验和计算都在预热隔离Pin worker完成，Kit主进程不加载Pin。

`results/actual_gravity_cpu_02`的21姿态×15上身势能差分误差≤3.99e−8Nm，另独立手写URDF FK对551项关节导数核验≤7.73e−9Nm。旧左腕pitch额外约−.233Nm补偿量消除后，实际末窗qref lead从约50mrad降至.34–.44mrad，持续卡antiwindup现象消失。双yaw速度和停车改善来自实际组合实测，不把静态质量验证等同整套动态验收；left与移动摄影停车仍需解决。

新的异步边界还绑定已接纳完整Task哈希和相机路径：旧Task的响应即使仍在原60ms时间内，也只触发保持/原slew制动和新观测请求，不跨Task执行，不把普通切换计作solver fault。当前Task的echo、seq、stamp和双时钟lease必须同时通过。`solver_feedback.py`让缺fresh/过期/普通superseded回报`command_stale`且feasible未知，真正QP无解、deadline、物理自碰撞分别明确失败。`qpdiag01`中保存了真正失败时Pink异常frame里的原Problem；不存在通过重建历史state伪造原始矩阵的做法。

`phase_lease01`安全中止后的独立同状态FD确认碰撞距离梯度误差≤3.43e−11m/rad，critical pair始终在最近20条约束中。原生QP给出的分离/制动速度满足原约束，但实测关节仍继续接近：26.98s原生命令距离率−.176mm/s、实测约−13.823mm/s。加入全部473条约束在9个可解状态中得到完全相同的命令，不能解决此执行跟随问题。`results/upper_damping_tracking_cpu_01`记录末段原15上身峰值力矩/cap仅.10711，没有饱和；D3→5的局部对角惯量敏感度估计改善仅约.12–.32mm，不能当成安全距离认证。后续物理对照保留原门槛与真正中止。

条件式`following_reserve.py`只作保留的实验选项，**不是已经验证的物理制动证书**。原60ms延迟、60ms额外前瞻与假设制动能力构造的附加20行约束，虽然通过真实Popen、canonical15↔PhysX29顺序、Task/seq/完整evidence哈希、真实已提交reference、planned与committed分离及4子步lease检查，首次闭环仍产生不可行。独立原矩阵IIS定位为新增torso↔left shoulder yaw、原torso↔left elbow与三条原加速度边界冲突：request2.56s新增约束要求20ms内分离17.099mm/s，原110行允许的最大分离约4.308mm/s。仅诊断去掉新增20行后原110行可解；该诊断没有执行放宽原距离或加速度。下一候选已关闭该条件模型，原6.5mm原生barrier、548对FCL和5mm物理门槛均保留。

`results/arm_stiffness_algebra_cpu_01`在原D5实测状态上比较臂Kp×2/×4，总PD+g/cap最大分别.1382/.1759，没有同状态裁剪；这不是未来瞬态或耦合稳定证明。新Kp×2物理结果通过启动前的实际armature及显式/引擎增益区分检查，但仍失败，未提升正式控制gate。腰Kp、12个原生learned腿输出及所有原物理范围保持。完整profile对照见`results/arm_stiffness2_profile_comparison_cpu_01`；原5s完整运动区间与最后完整.5s全部29关节窗口都没有移动。

`--g1-park-yaw-assist`为独立可选输入层：原ParkingSupervisor先完成PARK进入或25mm/50mrad退出判断；仅仍在PARK时，XY命令保持0，yaw使用`1.5*原anchor_yaw_error − .8*实测wz`，总输入上限±.08rad/s并遵守原.8rad/s² slew，没有积分器。原FCL与fresh/lease制动仍在它之后；实际输入被下游否决时完整留证。它不更改12个learned actions、机身状态、原WORLD相机或硬限位，也不认证checkpoint近零非零命令的训练分布。单左移物理结果支持抑制该处慢偏航，完整同版验收仍待完成。

`--g1-refresh-posture-on-park`保留减速开始时的软姿态保护，只在每个新Task首次原PARK进入事件，用同stamp实际joint_pos刷新一次15关节软偏好；随后退出/再次进入不会逐帧追随q，下一Task才重置。原waist yaw home0特殊软偏好、FrameTask目标/成本、CBF/范围不变，没有锁臂。`results/park_body_compensation_cpu_03`的三个真实终端J样本中，旧锚偏好对应native-null能量83.7–96.1%，与其下降梯度同向；真正FrameTask J投影交叉差≤2.9e−15。刷新后的实际结果与新分解`results/park_posture_refresh_decomposition_cpu_02`显示残余来源不同：摄影主要Frame-rowspace，右转仍有85.5% null能量，因此不能把全部余动归于机身XY漂移。该功能仍是待完整验收的可选候选。

`results/park_damping03_QP_gamma04_cpu_01`对D.03的四个实际异常帧保留原P/q/G和110行约束，仅缩放20条原生CBF的class-K右端项：gamma .1/s时四项LP/quadprog均无解，.4/s时四项均可解，最小所需gamma约.116–.365/s。原最近距离8.03–9.50mm、6.5mm原生floor和5mm实测门槛没有改动。较大gamma允许更快接近，原gamma1实测跟随滞后碰撞失败仍保留；CPU可行性不构成实际制动、停车或未来所有QP可行的保证。上表gamma .4实际对照仍有一个真正QP失败，不能把四个旧异常帧可解推广为全局保证。

`--g1-gravity-hold-profile fresh_current_state_v1`已纳入基础入口：运动包不新鲜时，每5ms从刚更新的SDK29q和pelvis实际旋转重新计算15上身关节的静态重力，只保留上一真实位置reference、速度目标0；不复用过期包的重力或速度，不延长60ms lease，不改变公共失败状态。NumPy/XML模型与原实际30body模型的20姿态×29项Pin对照最大误差4.44e−15Nm，独立合同21项检查通过；四项物理对照的8400次write另经独立审计，75次local静态支持与真正Pin一致。源模型、真实测量和每次write来源单独记录，原显式PD+g总和只经原cap一次限幅；模型吻合与实际方向/停车验证仍分别报告。

继承 rig 有4个LiDAR。此次没有改变扫描器数量/安装；局部RGB探针与上述运动不是全姿态无遮挡/避障充分感知认证，`sensors_avoidance_verified` 保持False。

官方固定来源：[WBC-AGILE](https://github.com/nvidia-isaac/WBC-AGILE)、[Pink](https://github.com/pink-kinematics/pink)、[Unitree URDF固定提交](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description/g1_29dof.urdf)。本地WBC-AGILE v1.2 commit `ad4ff6ee6b38da207592b3e71dcee4c118049f6d`，官方URDF provenance/原许可在 `kinematics/`。
