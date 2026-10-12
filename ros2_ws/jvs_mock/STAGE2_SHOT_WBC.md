# Stage 2:Shot-aware 预测式全身控制

参考:[Li et al., arXiv:2608.07005](https://arxiv.org/html/2608.07005v1)。

本阶段在 Stage 1 基础上增加 shot 分配与 configuration preview。它不实现
KC-SVSDF,也不声称具备避障能力。保留参考论文的 nonholonomic 状态与
forward/reverse 约定;shot 语义与相机任务代价是面向应用的新增内容。

## 模型与控制器

状态 `[x,y,yaw,q1..q6]`,输入 `[v,omega,dq1..dq6]`。精确的瞬时 nonholonomic
映射为 `xdot=cos(yaw)*v, ydot=sin(yaw)*v`;轮速为
`(v ± separation*omega/2)/radius`。

`shot_predictive_qp.py` 包含一个 receding-horizon LTV kinematic MPC 与一个
50 Hz constrained tracking QP。preview 为 8 个 0.4 s 区间(3.2 s),每 0.2 s
重规划,沿非线性 warm-start rollout 对耦合的九态动力学做线性化,并最小化相机
位姿误差、shot 相关运动代价、base heading 误差、机械臂参考偏差与控制差分。
travel 类 shot 惩罚偏离初始 pitch/elbow 姿态,以抑制不必要的伸直。该软代价
不保证奇异位形规避;残余的相机高度偏置计入误差统计。快速 QP 显式加权规划出的
base 控制量,使相机修正不撤销 heading 预备动作。preview 的首个全身控制量经
0.3 s reference filter 平滑后,作为快速 QP 的软参考。实测反馈闭环;不使用
pseudoinverse,也不做执行器裁剪。未来状态是局部线性预测,不是精确非线性或
物理可行性证明。

MPC 包含 base/joint 与轮速/轮加速度约束、实测与积分关节位置约束,以及一个
保守的分段线性关节制动包络。实测路径使用 URDF 边界;积分命令保留
position margin 作为伺服余量。执行的 QP 保留 Stage 1 针对实测/命令位置的精确
一步制动约束,以及命令加速度约束。

Orbit 偏好 base 平面运动与 forward 切向 heading。Pan 强惩罚 base 平移/旋转,
偏好 wrist 运动,并在 yaw 关节接近限位时放松 base 惩罚。MoveOut 偏好 base
平移,并在 MoveOut 首次进入 horizon 时通过同时求解两个 constrained MPC 候选
来选择 forward/reverse heading 分支,该选择按 segment 保留以避免方向抖动。
所有偏好都是软性的,从属于硬性执行器命令约束。shot 权重跨 2 s 过渡区间使用
cubic partition of unity。

## 输入

相对 6D 轨迹可包含一个 `shots` 数组,由连续区间组成:

```json
"shots": [
  {"type": "orbit", "start_s": 0, "end_s": 22},
  {"type": "pan", "start_s": 22, "end_s": 36},
  {"type": "moveout", "start_s": 36, "end_s": 54}
]
```

当前类型、phase `[0,1]` 与 next shot 在每个控制 tick 求值。旧的
`shot_metadata.shot` 契约仍支持单一 shot;未标注的 Stage 1 轨迹使用中性分配。

可选的 `/wbc/shot_context`(`std_msgs/String`)提供 JSON:
`{"shot_type":"pan","shot_phase":0.8,"next_shot":"moveout","remaining_s":2.0}`。
消息接收时刻作为剩余时长的锚点。这会改变 shot 偏好;相机位姿目标仍来自轨迹。
静态 shot 时间线是默认方式,并提供完整的未来序列。

## 验收协议(在成对评估之前声明)

所有运动均在 Isaac Sim 中以原始 URDF、PGS solver、1 kHz physics 与 50 Hz 控制
测量。每个成对 case 使用完全相同的几何、目标、初始关节与驱动参数。orbit case
以相机侧向朝向启动,使 base 初始朝向近似沿切线;任意初始朝向的重定向问题不被
藏进跟踪指标。还包括单一 Pan、近限位 Pan 以及 MoveOut 两个方向。

- 跟踪:Stage 1 门槛(20 mm RMS / 50 mm max,0.035 rad RMS / 0.0873 rad max)。
- 正常 Pan:base 总行程 <=20 mm,yaw 范围 <=3 度。
- Orbit:运动样本 heading RMS <=15 度,base 投影贡献为参考平面相机行程的
  正值且 >=60%。
- MoveOut:base 投影贡献 >=70%;两个选定方向均需执行。
- 命令限制:所有记录的 residual <=1e-6。原始 physics 限制按 Stage 1 继承的
  容差单独审计;违规不做平滑。
- 过渡(junction):检查 ±1 s 命令差(按执行器速度限值归一化)及 base
  位姿/速度连续性。任何归一化 step >0.05 即失败。
- 对比:相对 Stage 1,跟踪与 junction step 综合指标目标降低 >=10%,饱和不
  增加。零饱和 baseline 无法展示饱和降低,将如实报告。
- 预测:与 `no_preview` 对比(相同的连续 shot 权重);报告 Pan 最后 3.2 s
  (MoveOut 开始之前)的 heading 预备。junction 处 next-shot heading 误差
  降低 >=5 度即证明 preview 有效。

Stage 2 还提高了任务跟踪权重与反馈增益(4/s,Stage 1 为 3/s)。它与 Stage 1
的对比度量的是完整的控制器变化;no-preview 消融保留同样的 3.2 s horizon、
模型、代价权重、reference filter 与 5 Hz 更新率,冻结当前 shot 权重/类型并
外推当前相机 twist,而不是读取未来轨迹与 shot 样本,以此隔离"能否获取未来
shot/轨迹信息"这一因素。仅通过精度门槛不构成 Stage 2 的整体验收结论。

## 复现

在 `ros2_ws` 下,source ROS 与本 workspace:

```bash
/usr/bin/python3 jvs_mock/scripts/generate_shot_trajectories.py
colcon build --symlink-install --packages-select jvs_mock
source install/setup.bash
export PYTHONPATH=/tmp/jvs_wbc_deps:$PYTHONPATH
OMNI_KIT_ACCEPT_EULA=yes /usr/bin/python3 jvs_mock/scripts/test_shot_suite.py \
  --output /tmp/stage2_paired
OMNI_KIT_ACCEPT_EULA=yes /usr/bin/python3 jvs_mock/scripts/test_shot_suite.py \
  --output /tmp/stage2_ablation --cases combo1 --controllers no_preview
ros2 launch jvs_mock jvs_shot_wbc_isaac.launch.py headless:=true \
  trajectory_file:=/absolute/path/to/shot_pan.json output_directory:=/tmp/shot_ros
```

launch 参数 `controller_mode` 支持 `stage1`、`stage2`、`no_preview`。lockstep
runner 还接受 `--initial-q` 以使用 manifest 中受控的初始构型;标准 ROS launch
从仿真器的常规零关节位形启动。

输出包括独立的 PhysX 相机位姿/目标、位置/SO(3) 误差、base 与关节的实测/命令
速度、shot 权重、phase、heading 分支、preview 参考、预测末端 heading、QP
residuals,以及原始 1 ms physics 审计。报告的 MPC 耗时包含构建与求解;
fast QP 耗时单独记录。

lockstep headless runner 以 5 Hz 泵 Kit 渲染,与 1 kHz physics/审计及 50 Hz
相机反馈相互独立。它在每个控制区间后检查 physics 时钟。ROS 后端保持其既有的
发布节奏。

输入契约检查(非机器人验收测试):
`PYTHONPATH=/tmp/jvs_wbc_deps /usr/bin/python3 -m unittest discover -s jvs_mock/test -p test_shot_contract.py`。

MPC 仅在通过独立的 primal-residual(<=1e-6)与归一化 stationarity(<=1e-5)
检查后才接受一个 plan。边缘求解会以更紧的 OSQP 容差重试;不使用执行器或解的
裁剪。`solved` 与 `solved inaccurate` 都受这些检查约束。迭代数、重试数、
residual 与总更新耗时均被记录。

ROS Stage 2 节点在派生的 worker 进程中运行 MPC。快速反馈 QP 从不等待 preview
求解;plan 安装会检查 shot 输入代次与 0.6 s 仿真时间龄期限制。preview 龄期、
在途状态与被丢弃的过期 plan 计数均有日志。lockstep 测试同步调用完全相同的
planner。

## 结果(Isaac Sim,2026-09-18)

最终成对报告与图:[results/stage2_report/REPORT.md](results/stage2_report/REPORT.md)。
被接受的目录与被排除的开发运行清单:[PROVENANCE.md](results/stage2_report/PROVENANCE.md)。
示例与初始条件:[shot_stage2_manifest.json](trajectories/shot_stage2_manifest.json)。
preview 参数:[shot_whole_body_controller.yaml](config/shot_whole_body_controller.yaml)。

六个 Stage 1 baseline、六个最终 Stage 2 case 与匹配的消融全部完成,并通过
继承的跟踪门槛。以下误差包含保持段与预备段,不排除过渡成本。

| Case | 位置 RMS mm,Stage 1 → 2 | 姿态 RMS deg,Stage 1 → 2 |
|---|---:|---:|
| Orbit | 0.471 → 2.070 | 0.160 → 0.154 |
| Pan | 0.619 → 0.079 | 0.161 → 0.007 |
| Pan near limits | 0.769 → 0.155 | 0.170 → 0.151 |
| MoveOut reverse | 0.137 → 0.043 | 0.009 → 0.005 |
| MoveOut forward | 1.756 → 0.301 | 0.240 → 0.201 |
| Orbit → Pan → MoveOut | 5.666 → 4.576 | 0.644 → 0.206 |

Orbit 运动样本的切向 heading RMS 为 2.28 度。与 Stage 1 相比其位置误差增大,
但 base heading 跟随了 shot 意图。正常 Pan 行程 10.8 mm,base yaw 范围仅
0.00023 度;近限位 Pan 允许 24.54 度 base yaw,并在全部命令限制内完成。
MoveOut 两个方向分支均被执行;base 投影平移约占相机平移的全部。投影占比略
大于 1 是少量反向机械臂补偿所致,并非额外的路径运动。

组合 case 将位置/姿态 RMS 分别降低 19.2%/68.0%,位置峰值从 22.41 降至
9.60 mm,全序列最大归一化命令 step 从 0.03711 降至 0.01712(53.9%)。后期
MoveOut 中残余的约 7–8 mm 高度偏置计入上述统计;posture regularization 是
软性的。

在 Pan 结束之前,preview 使 base heading 改变 16.73 度,匹配消融仅为
0.004 度。junction 处 next-shot heading 误差为 27.70 对 42.72 度。preview
还把消融的 junction step 从 0.02241 降至 0.00933。它**并非**改善所有指标:
带 preview 的位置 RMS 为 4.576 mm,不带为 4.350 mm,而姿态 RMS 从约 0.246
改善到 0.206 度。因此该预备动作有可测的位置精度代价。

完整的异步 ROS launch 也通过 Pan 跟踪:位置 RMS 0.089 mm、姿态 RMS
0.016 度,无控制器 fault,控制器、preview worker 与 Isaac 均干净关闭。这是
控制率 FK 验证;lockstep case 提供独立的 PhysX 相机位姿与 1 ms 约束审计。
构建、三个输入契约测试、Python 编译与 diff 空白检查均通过。

**Stage 2 整体验收:未完全通过。**分配、自适应 Pan、diff-drive heading、
有界的 junction 连续性、组合跟踪聚合,以及有效的预测预备,均通过既定检查。
全部命令/预测 residual 在 1e-6 内;最终 case 中实测位置、速度与轮速保持在
限值内。但是:

- 原始 1 ms physics 审计仍发现 base/joint 或轮加速度违规;继承的严格物理
  约束要求未满足。
- Stage 1 的静止 junction 保持段命令 step 本已近乎为零。预备运动使 junction
  窗口 step 增至 0.00933,虽仍低于连续性门槛 0.05,但未展示相对 Stage 1 的
  降低。
- 两个控制器在最终组合中命令速度/加速度饱和均为零。展示了"不增加";
  无法声称"降低"。

不声称任何避障或 KC-SVSDF 验收结论。

重新生成完整报告(在 `ros2_ws` 下,依赖可用时):

```bash
PYTHONPATH=/tmp/jvs_wbc_deps MPLCONFIGDIR=/tmp/jvs_mpl /usr/bin/python3 \
  jvs_mock/scripts/summarize_shot_results.py \
  jvs_mock/results/stage2_final jvs_mock/results/stage2_paired \
  --output jvs_mock/results/stage2_report --ros jvs_mock/results/stage2_ros_verified
```
