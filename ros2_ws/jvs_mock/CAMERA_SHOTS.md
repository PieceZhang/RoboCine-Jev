# 常规运镜轨迹示例与 Isaac 跟踪测试

本轮重点是 orbit、pan、moveout 的 6D 相机轨迹跟踪精度，不运行极限压力测试。
控制器仍为原来的 constrained QP，保持默认 PGS、伺服增益和执行器限值。

| 文件（均位于 trajectories/） | 运镜 | 参数 | 总时长 / 运动时间 |
|---|---|---|---|
| [wbc_orbit_ccw10.json](trajectories/wbc_orbit_ccw10.json) | Orbit | 半径 0.6 m，逆时针 10° | 24 / 20 s |
| [wbc_orbit_cw15.json](trajectories/wbc_orbit_cw15.json) | Orbit | 半径 0.8 m，顺时针 15° | 28 / 24 s |
| [wbc_pan_left20.json](trajectories/wbc_pan_left20.json) | Pan | 向左摇摄 20°，相机位置固定 | 16 / 12 s |
| [wbc_pan_right30.json](trajectories/wbc_pan_right30.json) | Pan | 向右摇摄 30°，相机位置固定 | 20 / 16 s |
| [wbc_moveout_040.json](trajectories/wbc_moveout_040.json) | Moveout | 沿视线反方向后退 0.4 m | 16 / 12 s |
| [wbc_moveout_080.json](trajectories/wbc_moveout_080.json) | Moveout | 沿视线反方向后退 0.8 m | 24 / 20 s |

采用 `jvs_relative_task_space_trajectory/v1` 格式，参考系为 `initial_camera_link`。
相机初始 +X 为视线方向、+Y 为左、+Z 为上；这里使用 camera_link，不是 optical frame。
所有文件指定完整 xyz/RPY 目标，未运动的自由度也要保持。

Orbit 的固定主体为初始相机坐标中的 `[R,0,0]`：

```text
p(theta) = [R(1-cos(theta)), -R sin(theta), 0]
R_camera(theta) = Rz(theta)
```

因此主体始终位于参考相机 +X 视线方向，轨迹是定半径圆弧。
Pan 的位置保持不变，只改变 yaw；Moveout 的姿态保持不变，位置沿 -X 改变。
轨迹直接定义于相机空间，未向 WBC 提供指定的底盘或关节运动分配。

每条轨迹首尾各静止 2 s，运动区间采用五次 smoothstep：`s=10τ³-15τ⁴+6τ⁵`。
其起终点速度和加速度为零；JSON 按 20 Hz 采样，WBC 使用平移样条和 SO(3) 样条插值。

## 生成、启动和测试

在工作区根目录运行：

```bash
/usr/bin/python3 ros2_ws/jvs_mock/scripts/generate_camera_shots.py
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --packages-select jvs_mock --symlink-install
source install/setup.bash
# 系统 Python 需 numpy、scipy、osqp；本次隔离依赖位于 /tmp/jvs_wbc_deps。
export PYTHONPATH=/tmp/jvs_wbc_deps:$PYTHONPATH
# 已接受 NVIDIA EULA 后设置：
export OMNI_KIT_ACCEPT_EULA=yes
ros2 launch jvs_mock jvs_wbc_isaac.launch.py headless:=true \
  trajectory_file:=$PWD/jvs_mock/trajectories/wbc_orbit_ccw10.json \
  output_directory:=/tmp/wbc_orbit_ros
```

完整六条轨迹的独立 Isaac 物理测试（在 `ros2_ws` 中）：

```bash
/usr/bin/python3 jvs_mock/scripts/test_wbc_suite.py \
  --isaac-python /home/w84384651/miniconda3/envs/env_isaaclab/bin/python \
  --output /tmp/wbc_camera_shots --criterion tracking
MPLCONFIGDIR=/tmp/wbc_plot_cache /usr/bin/python3 \
  jvs_mock/scripts/summarize_camera_shots.py /tmp/wbc_camera_shots
```

默认只运行上述六条常规轨迹；也可用 `--cases orbit_ccw10 pan_left20 moveout_040`
选择子集。汇总脚本当前要求完整六条结果。
测试使用空场景、真实 URDF 刚体与地面接触，physics=1000 Hz、QP=50 Hz。
相机实际位姿来自 PhysX camera_link，不以控制器 FK 代替实际轨迹。

## 结果判定

主要关注运动区间的相机位置误差与 SO(3) 姿态误差，报告 RMS、P95、最大值，
并另外保留包含首尾静止段的统计。使用此前精度门槛：位置 RMS 20 mm、最大 50 mm；
姿态 RMS 0.035 rad、最大 0.0873 rad。
Pan、Moveout 不强制 base 与 arm 都移动；只要相机正确跟踪即可。

`--criterion tracking` 根据精度和执行完成情况返回 0/2；
`--criterion stage1` 仍使用此前包含全部物理约束的严格判定。
两者不会混淆：普通跟踪精度通过，不代表逐物理步加速度约束全部通过。
底层 `summary.json` 和 `physics.csv.gz` 中仍保留原始约束诊断。

每个测试目录新增 `camera_tracking.png`、`tracking_accuracy.json`，
`telemetry.csv` 同时记录目标与实际相机 xyz/quaternion、误差、base v/ω 和关节速度。

本次实测结果见 [常规运镜报告](results/camera_shots_20260918/REPORT.md)。
