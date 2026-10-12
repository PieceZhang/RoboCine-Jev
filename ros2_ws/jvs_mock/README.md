# jvs_mock

RoboCine 机器人的自包含 ROS 2 Humble mock，支持 Gazebo Fortress 与
NVIDIA Isaac Sim 5.1 两个仿真后端。模型主要使用 URDF 基础几何体，并为底盘使用
一个本地凸圆台 STL：直径 0.14 m 的左右驱动轮、圆柱化
6-DOF 机械臂、末端相机、独立的机身 RGBD 相机和机身 3D LiDAR。为避免理想两轮模型在物理
仿真中绕轮轴倾倒，底盘设有一只前中万向轮和两只后万向轮；两个后端的
接地中心为 `(0.180, 0, 0)` m、`(-0.131, ±0.089, 0)` m，轮半径 0.02 m，
与主动轮共面。每个万向轮都有自由转向和滚动关节；Gazebo 直接使用被动关节，
Isaac 显式将导入器创建的驱动设为零力，避免锁死万向轮。支撑座、机械臂安装台
和 RGBD 支架均有简单碰撞体及惯量。
安装台补齐底盘顶面与机械臂安装点之间的 0.076 m 间隙。两个后端使用完全相同的
连杆、关节、碰撞体、质量和惯量，不使用后端专用的几何偏移。

底盘主体为上半径 0.13 m、下半径 0.18 m、高度 0.12 m 的圆台，质量为
`31.772322 kg`。扩大的下缘与三个万向轮转向座相交，使支撑结构在视觉和
碰撞几何上都连接到底盘；转向座同时向上延伸至底盘内部。
按需求保留了现有主体高度和 `z=0.13 m` 的几何中心，因此主体顶面仍约为
`z=0.19 m`。均匀圆台的惯性中心按实际形状下移至约 `z=0.1236 m`；轮距、
轮径、左右主动轮位置和控制接口保持不变。
颈部和六节机械臂质量统一乘以 `arm_mass_scale=0.6`，
由合计 2.17 kg 降至 1.302 kg，末端相机仍为 0.12 kg。惯量由相同几何公式随质量
同步重算。Isaac 使用 URDF 的惯量及各关节 effort 上限（轮子 20 N·m，机械臂
依次为 10/10/10/8/8/6 N·m），位置驱动显式使用 force 模式。

两个后端的主动轮摩擦系数均为 1.0，万向轮为 0.02。Isaac 显式采用 `min` 摩擦
组合规则，并把默认地面摩擦设为 1.0，避免默认平均规则提高万向轮接触摩擦。
相关规则见 [PhysX 材质文档](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html)。
外部 USD 场景的地面材质仍需单独核对。默认物理步长统一为 1 ms，重力为 9.81 m/s²；
可用 `physics_hz` 调低 Isaac 频率以提高性能，但会改变求解精度。

底盘均为浮动基座，零速度命令只控制轮速，不锁定底盘的位置或姿态。两者命令更新
频率均为 100 Hz，速度上限为 1 m/s、3 rad/s，加速度上限为 1 m/s²、7 rad/s²。
Gazebo 的 ros2_control 速度接口与 Isaac 的有限力矩速度伺服仍有动力学差异，
接触求解器也不同；统一参数不等于动态响应完全相同。最大伸展轨迹应在两端回放，
检查底盘俯仰/横滚、位移、实际轮速和主动轮是否空转。

轮距/轮径、原有机械臂关节变换/限位、相机外参及控制接口保持不变。
回归测试检查两端展开模型一致，并检查关节限位角点及随机姿态的静态支撑余量
大于 25 mm。一前两后三只从动轮扩大了前后倾覆稳定范围，
但不保证急加减速、斜坡或任意负载下不会倾倒；仿真回放时仍需检查实际接触。

## 文件

- `urdf/jvs_mock.urdf.xacro`：连杆、关节、惯量、碰撞体及相机/LiDAR 坐标系。
- `urdf/ros2_control.xacro`：轮速和六关节位置 command/state interfaces。
- `urdf/gazebo.xacro`：`gz_ros2_control` 系统插件、`/cmd_vel` remap 与 GPU LiDAR。
- `config/jarvis_controllers.yaml`：底盘、手臂和 joint-state 控制器。
- `launch/jvs_mock_gazebo.launch.py`：Gazebo、模型、时钟桥及控制器 bringup。
- `launch/jvs_mock_isaac.launch.py`：Isaac Sim、ROS 2 Bridge 与命令适配器 bringup。
- `launch/jvs_mock_simview.launch.py`：在外部 USDA/USD 场景中加载 JVS 与相机轨迹。
- `scripts/jvs_isaac_sim.py`：运行在 Isaac Python 内的 SimulationApp/PhysX runner。
- `scripts/jvs_simview_isaac.py`：读取 Y-up 或 Z-up USDA/USD 并组合为 Isaac Z-up stage。
- `jvs_mock/isaac_command_adapter.py`：保持两个后端的 ROS 命令接口一致。
- `launch/joint_space_test.launch.py`：六关节缓慢正弦位置指令。
- `launch/jvs_follow_traj.launch.py`：读取相对 task-space JSON、IK 解算并回放。
- `launch/jarvis_tf_rviz.launch.py`：RViz 可视化运行中仿真的 TF 树。
- `config/jarvis_tf.rviz`：上述 launch 使用的 RViz 配置（TF + RobotModel +
  `/cam_desired_traj` 路径 + `path_markers`）。
- `trajectories/default_task_space_trajectory.json`：10 Hz、6 秒、30 cm/20° 默认轨迹。

## 环境和构建

新增的动态 LiDAR KC-WBC 入口和三个报告场景见 [动态避障使用报告](DYNAMIC_WBC.md)。
`jvs_kc_wbc_dynamic_trajfollow.launch.py` 支持相机轨迹与障碍物轨迹文件，
默认依次运行全部现有 combo。原有 trajfollow 保持不变。

```bash
sudo apt install ros-humble-ros-gz ros-humble-gz-ros2-control \
  ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-ros2topic

cd RoboCine/ros2_ws
source /opt/ros/humble/setup.bash
# 若本机从未初始化 rosdep，先执行：sudo rosdep init && rosdep update
rosdep install --from-paths jvs_mock --ignore-src -r -y --rosdistro humble
colcon build --packages-select jvs_mock
source install/setup.bash
```

## 启动 Gazebo 与控制器

```bash
ros2 launch jvs_mock jvs_mock_gazebo.launch.py

# 无图形界面的服务器（适用于 SSH、CI 或没有可用显示器的环境）
ros2 launch jvs_mock jvs_mock_gazebo.launch.py gui:=false
```

启动顺序是模型生成后加载 `joint_state_broadcaster`，然后配置并激活
`base_controller` 和 `arm_position_controller`。终端出现 `Controllers active`
表示三个控制器均已激活。Gazebo 服务器和图形客户端由独立进程启动，以避免
Fortress 的组合启动器在 GUI 握手失败时永久等待 world list。启动文件默认将
Gazebo Transport 绑定到 `127.0.0.1`，避免 VPN、Docker 或断开的网卡使本机进程
互相无法发现；确实需要跨机器仿真时可用 `gz_ip:=<本机局域网IP>` 覆盖。

Gazebo 后端默认启用安装在 `base_link` 上的前后两个 GPU LiDAR。后雷达为 64 线，位于
底盘后侧顶面，避开前侧 RGBD 与机械臂底座；`lidar_link` 外参为
`xyz=(-0.085, 0, 0.235) m`、`rpy=(0, 0, 0) rad`。扫描参数为水平
360°/360 samples、垂直 -45–45°/64 samples、15 Hz、量程 0.12–50 m、
距离分辨率 0.01 m。
Gazebo Transport 使用 `/lidar`，ROS 2 仅单向接收
`/lidar/points` (`sensor_msgs/msg/PointCloud2`)；二维 LaserScan 不桥接。
前雷达恢复原始外参 `xyz=(0.190, 0, 0.274)`、`rpy=(0, 0.898844565, 0)`，
坐标系为 `front_lidar_link`，Gazebo 原始垂直扫描为 0–60°/96 samples；
点云桥接至 `/lidar/front/points`，不与后雷达共用话题。
外观使用包内的简化多面体 `meshes/jarvis_lidar.stl`，STL 以毫米建模并在
URDF 中按 `0.001` 缩放为米；mesh 原点与 `lidar_link` 原点重合。
使用自定义 world 时也必须加载 Gazebo Sensors system（默认
`worlds/empty.world` 已配置）。

## 启动 Isaac Sim

Isaac runner 使用 Isaac Sim 5.1 的 Python 环境，并通过 OmniGraph ROS 2 Bridge
直接控制 PhysX articulation。先确认 `nvidia-smi` 正常，再执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=yes  # 仅在已经接受 NVIDIA Omniverse EULA 后设置

cd RoboCine/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

# Isaac GUI
ros2 launch jvs_mock jvs_mock_isaac.launch.py

# 无界面运行
ros2 launch jvs_mock jvs_mock_isaac.launch.py headless:=true
```

launch 会自动优先使用 `$ISAAC_PYTHON`、当前已安装 Isaac Sim 的 Conda
环境或常见路径 `~/miniconda3/envs/env_isaaclab/bin/python`。因此即使当前是
Conda `base` 环境，也不会遮蔽 `env_isaaclab`。非标准安装位置可显式指定：

```bash
ros2 launch jvs_mock jvs_mock_isaac.launch.py \
  isaac_python:=/absolute/path/to/env_isaaclab/bin/python
```

可用 `stage:=/absolute/environment.usd` 打开已有 Isaac 场景；默认创建空场景和
地面。`save_stage:=/absolute/jvs_scene.usda` 可保存导入机器人和 OmniGraph 后的
组合 stage。默认只启动仿真，若要同时回放 task-space 轨迹：

```bash
ros2 launch jvs_mock jvs_mock_isaac.launch.py \
  start_trajectory_follower:=true \
  trajectory_file:=/absolute/path/to/trajectory.json
```

Isaac 后端在启动时以 `sim_backend:=isaac` 展开 Xacro，因此不会载入
`gz_ros2_control` 或 Gazebo 接触标签。runner 导入 URDF、配置轮子的速度 drive、
六关节的位置 drive，并创建前后两个 RTX LiDAR。后雷达在 `lidar_link` 下，默认使用
包内的 `config/JVS_Nearfield.json`：0.10 m 近距、30 m 远距、360° 旋转扫描、
垂直 -45–45° 的 64 个通道。完整扫描以
`sensor_msgs/msg/PointCloud2` 发布到 `/lidar/points`，消息 `frame_id` 为
`lidar_link`，时间戳使用仿真时间。普通 Isaac launch 和 simview/Kujiale launch
使用相同的 LiDAR 实现与默认值。
前雷达在 `front_lidar_link` 下，恢复原始 `Example_Rotary` 配置及俯仰外参，
发布至 `/lidar/front/points`；不是复制后雷达的近场配置。两个 frame 均由 TF 发布。
双雷达离线融合及场景三角 mesh 重建见 [DUAL_LIDAR_MESH.md](DUAL_LIDAR_MESH.md)。

runner 建立如下桥接：

- `/cmd_vel` 与 `/arm_position_controller/commands` → 命令适配器（限幅、超时及差速换算）
  → `/isaac_joint_commands` → 单一 articulation 控制器；
- articulation → `/joint_states`、`/base_controller/odom`、`/tf`；
- 后/前 RTX LiDAR → `/lidar/points`、`/lidar/front/points`；
- Isaac 仿真时间 → `/clock`。

因此 `joint_space_test.launch.py` 和 `jvs_follow_traj.launch.py` 无需区分仿真后端。

LiDAR 常用覆盖参数如下：

```bash
ros2 launch jvs_mock jvs_mock_isaac.launch.py \
  lidar_config:=JVS_Nearfield \
  lidar_topic:=/lidar/points \
  lidar_full_scan:=true

# Kujiale/simview 使用同一组参数；排障时可显示 RTX 点云
ros2 launch jvs_mock jvs_mock_simview.launch.py \
  simview_stage:=/absolute/path/to/kujiale.usda \
  lidar_debug_view:=true
```

`enable_lidar:=false` 关闭两台传感器；`enable_front_lidar:=false` 仅关闭前雷达，
保留后雷达。`front_lidar_config`、`front_lidar_topic` 可覆盖前雷达配置/话题。
`lidar_mode:=front` 仅开启前雷达，`lidar_mode:=both` 同时开启前后雷达（默认）。
Isaac、simview 和 Gazebo 启动文件都支持该参数。示例：

```bash
ros2 launch jvs_mock jvs_mock_simview.launch.py \
  simview_stage:=/absolute/path/to/kujiale.usda lidar_mode:=front
ros2 launch jvs_mock jvs_mock_simview.launch.py \
  simview_stage:=/absolute/path/to/kujiale.usda lidar_mode:=both
ros2 launch jvs_mock jvs_mock_gazebo.launch.py lidar_mode:=front
```

旧参数 `enable_front_lidar:=false` 在 Isaac/simview 中仍保留“仅后雷达”行为；
它与 `lidar_mode:=front` 同时使用会报配置冲突。`enable_lidar:=false` 仍关闭全部雷达。
`lidar_variant` 可选择后雷达带 variant 的 Isaac
传感器型号；`lidar_full_scan:=false` 会按每个渲染帧发布部分扫描，降低延迟但改变
点云分包方式。`lidar_motion_bvh:=true` 可模拟扫描期间机器人/物体运动造成的光线
变化，但 RTX Motion BVH 会显著增加启动时间和显存；本机 32 GB GPU 在存在其他
GPU 负载时运行 Kujiale 0021 的测试曾超出显存，因此默认关闭。关闭时传感器位姿
仍随 `lidar_link` 更新，只是不模拟单次扫描内部的运动效应。

`JVS_Nearfield` 是面向近场避障的理想化旋转式 LiDAR，并不是论文中的 Livox
MID-360 精确扫描模型；要做传感器级复现，还需补充 MID-360 的发射/扫描 pattern
配置以及下游需要的逐点时间字段。当前接口已把型号选择与 ROS 话题隔离，后续替换
配置不需要修改机器人外参或启动链路。

底盘接口为 `/cmd_vel` (`geometry_msgs/msg/Twist`)。必须持续发布；0.5 秒没有
新消息后目标速度归零，并按加速度限制减速停止：

```bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.15}, angular: {z: 0.20}}'
```

## 在 USDA/simview 场景中启动 Isaac Sim

`jvs_mock_simview.launch.py` 用于把现有 USDA/USD 场景作为 reference 挂载到
`/World/Simview`，然后导入 JVS、创建机器人光学相机及 ROS 2 OmniGraph。完成上面的
Isaac 环境配置和 workspace source 后，以 USDA/USD 的绝对路径启动：

```bash
ros2 launch jvs_mock jvs_mock_simview.launch.py \
  simview_stage:=/absolute/path/to/scene.usda
```

例如，直接读取原始 InteriorAgent 场景：

```bash
ros2 launch jvs_mock jvs_mock_simview.launch.py \
  simview_stage:=/home/w84384651/Documents/RoboCine/datasets/interioragent/kujiale_0021/kujiale_0021.usda
```

也可以读取其 Y-up wrapper，或 RoboCine 生成的 `simview.usda`：

```bash
# Y-up wrapper
ros2 launch jvs_mock jvs_mock_simview.launch.py \
  simview_stage:=/absolute/path/to/kujiale_0021_yup.usda

# RoboCine 输出
ros2 launch jvs_mock jvs_mock_simview.launch.py \
  simview_stage:=/absolute/path/to/cinema_traj_out/shot_name/simview/simview.usda
```

读取场景时会优先使用 USDA 声明的 `defaultPrim`；没有声明时依次尝试 `/World` 和
唯一的 active root prim。程序同时支持 Y-up 与 Z-up：Y-up 场景会自动绕 X 轴旋转
`+90°` 转换为 Isaac 的 Z-up 坐标，原生 Z-up 场景保持不旋转。因此原始
InteriorAgent USDA 和 Y-up `simview.usda` 可以使用同一个 launch，无需手工改轴。
启动日志会打印实际选择，例如：

```text
JVS simview: mounted source prim=/Root, up_axis=Z, rotate_x=0
JVS simview: mounted source prim=/World, up_axis=Y, rotate_x=90
```

USDA 内的相对 reference/payload 路径由 USD 相对于源文件解析。移动数据集时应保留
原目录结构（例如主 USDA 旁的 `Meshes/*.usd`）；只复制顶层 USDA 可能造成场景缺失。

机器人默认出生在 kujiale_0021 客厅中央的开阔区域 `(x=5.0, z=1.0)`。出生位置使用
RoboCine 的 Y-up 坐标：`x`、`z` 表示平面位置，`yaw` 单位为弧度且 `0` 朝向
`+Z`。加载其他场景时可以覆盖位置，或先关闭机器人相机视角查看全局场景：

```bash
ros2 launch jvs_mock jvs_mock_simview.launch.py \
  simview_stage:=/absolute/path/to/scene.usda \
  x:=5.0 z:=1.0 yaw:=0.0

# 黑屏排查：不自动切换到机器人相机
ros2 launch jvs_mock jvs_mock_simview.launch.py \
  simview_stage:=/absolute/path/to/scene.usda \
  activate_camera:=false
```

常用参数如下：

- `headless:=true`：无界面运行。
- `enable_lidar:=true lidar_topic:=/lidar/points`：启用 RTX LiDAR 点云（默认）。
- `save_stage:=/absolute/path/to/composed.usda`：保存组合后的 Isaac stage。
- `start_trajectory_follower:=false`：不自动回放默认机械臂轨迹；此 launch 默认为
  `true`。
- `add_ground_plane:=true ground_visible:=false`：添加隐藏的物理地面；两项均为默认值。
- `viewport_scroll_scale:=0.2`：普通滚轮缩放/推拉灵敏度，Isaac 默认值为 `1.0`。
- `viewport_drag_scale:=1.0`：平移、环绕、右键观察和拖拽推拉的鼠标灵敏度。
  Isaac 默认值为 `1.0`；KC-WBC 场景启动入口默认使用 `0.2`。
- `viewport_speed_step:=1.02`：按住鼠标右键滚轮调整相机速度时的单步倍率，Isaac
  默认值为 `1.1`。KC-WBC 场景启动入口分别默认使用滚轮 `0.05`、拖拽 `0.2`
  和速度档位 `1.01`；若需要可在 launch 命令中覆盖这三个参数。

## RViz 可视化 TF 树

仿真（任一后端）运行时，另开一个已 source 的终端启动 RViz 查看实时 TF：

```bash
ros2 launch jvs_mock jarvis_tf_rviz.launch.py
```

该 launch 不启动仿真，只补齐 RViz 需要而仿真不发布的三块内容：
`robot_state_publisher`（提供 `/robot_description` 给 RobotModel 显示，
URDF 以 `sim_backend:=isaac` 展开以匹配 simview 模型）、静态恒等变换
`map → world`（planner 话题 `/cam_desired_traj` 与 `/cinema_traj_bridge/
path_markers` 使用 `map` 系，按 traj_server 的 Y-up → Z-up 轴序转换后与
仿真 `world` 系坐标一致，但没有节点发布该变换），以及加载
`config/jarvis_tf.rviz` 的 RViz 本身。TF 树（`world → base_link →
link1..link6 → camera_link → camera_optical_frame`、各主动轮/万向轮、
`odom → base_link`）由仿真在 `/tf` 上以约 90 Hz 发布，Fixed Frame 为
`world`。可传 `rviz_config:=<path>` 换用其他配置。

注意：若从 snap 版 VS Code 内置终端启动，VS Code 泄漏的 `GTK_PATH` /
`GTK_EXE_PREFIX` 等环境变量会让 rviz2 的 Qt gtk3 主题插件加载 snap
core20 的 glibc 而崩溃（`__libc_pthread_init ... GLIBC_PRIVATE`）。
本 launch 已用 `UnsetEnvironmentVariable` 自动清除这些变量；直接运行
`rviz2` 时需先 `unset GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE_FILE
GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR`。

## 验证六关节控制

保持仿真运行，在另一个已 source 的终端执行：

```bash
ros2 launch jvs_mock joint_space_test.launch.py
# 可选：更慢且幅度减半
ros2 launch jvs_mock joint_space_test.launch.py period_s:=16.0 amplitude_scale:=0.5
```

节点以 20 Hz 向 `/arm_position_controller/commands` 发布完整的
`Float64MultiArray`，顺序固定为 `joint1..joint6`。所有 ROS 节点均使用仿真时间；
Gazebo 的 `/clock` 来自 `ros_gz_bridge`，Isaac 的 `/clock` 来自 OmniGraph ROS 2 Bridge。

## 跟随相对 task-space 轨迹

保持仿真和控制器运行，另开终端启动（不要同时运行 `joint_space_commander`）：

```bash
ros2 launch jvs_mock jvs_follow_traj.launch.py

# 使用自己的本地 JSON
ros2 launch jvs_mock jvs_follow_traj.launch.py \
  trajectory_file:=/absolute/path/to/my_relative_trajectory.json
```

`jvs_follow_traj` 等待 `/joint_states` 中出现完整的 `joint1..joint6`，将此刻
`camera_link` 位姿锁定为相对轨迹原点。它先用带关节限位的阻尼最小二乘 IK
解算整条轨迹；任一点不可达时不会向 controller 发送部分轨迹。全部成功后，
按 JSON 时间戳向 `/arm_position_controller/commands` 发布六关节位置，并在终端
逐条打印实际发送的 `joint1..joint6` 指令；结束时保持最终指令。如不需要打印，
可在启动时设置 `log_commands:=false`。

默认文件含 61 个点（含首尾），每 0.1 秒一个点：相对初始位姿的最大平移距离为
30 cm，同时俯仰角变化到 -20°。该轨迹根据机械臂运动学生成并经过逐点 IK 验证。
JSON 格式如下：

```json
{
  "format": "jvs_relative_task_space_trajectory/v1",
  "relative_to": "current_camera_link",
  "frame_id": "base_link",
  "dof_order": ["x", "y", "z", "roll", "pitch", "yaw"],
  "units": {"translation": "m", "rotation": "rad"},
  "sample_rate_hz": 10.0,
  "duration_s": 6.0,
  "samples": [
    {"time_from_start": 0.0, "pose": [0, 0, 0, 0, 0, 0]},
    {"time_from_start": 0.1, "pose": [0.005158, 0, 0.001852, 0, -0.005818, 0]}
  ]
}
```

`pose` 顺序固定为 `[x, y, z, roll, pitch, yaw]`。首点必须在 `t=0` 且六项
全为零。`frame_id: "base_link"` 表示位移和相对旋转在底盘坐标系表达；也可用
`"initial_camera_link"`，此时六维增量在启动瞬间的相机坐标系表达。默认
`control_rate_hz:=0.0` 会使用 JSON 中的 `sample_rate_hz`；也可在 launch 参数中
覆盖发布频率，节点会按时间戳在相邻 joint-space 解之间插值。

可用以下命令检查接口：

```bash
ros2 control list_controllers
ros2 control list_hardware_interfaces
ros2 topic echo /joint_states
ros2 topic echo /base_controller/odom
```

## Whole-body QP camera tracking

新增 Stage 1 控制器、Isaac headless 验证及运行说明见 [STAGE1_WBC.md](STAGE1_WBC.md)。
启动入口：`ros2 launch jvs_mock jvs_wbc_isaac.launch.py headless:=true`。
实际测试结果和未通过项见 [Stage 1 验收报告](results/REPORT.md)。

常规 **orbit / pan / moveout** 示例、精度测试与运行方式见
[CAMERA_SHOTS.md](CAMERA_SHOTS.md)。默认测试套件已改为六条常规运镜，重点评估相机跟踪精度。
## Shot-aware predictive WBC

Stage 2 adds Orbit/Pan/MoveOut allocation, continuous shot weights and a 3.2 s
whole-body preview. See [STAGE2_SHOT_WBC.md](STAGE2_SHOT_WBC.md) for the model,
input contract, paired Isaac tests and acceptance criteria. Start the normal
Pan demo with `ros2 launch jvs_mock jvs_shot_wbc_isaac.launch.py headless:=true`.
KC-SVSDF and obstacle avoidance are not included in this stage.

Stage 3 的实验性 KC-SVSDF 预测 WBC、静态双障碍场景和三种模式的 Isaac 对比见
[STAGE3_KC_WBC.md](STAGE3_KC_WBC.md)。`jvs_kc_wbc_isaac.launch.py` 只负责
启动 Isaac 和指定场景；另一个终端中的 `jvs_kc_wbc_trajfollow.launch.py` 负责
先把 Isaac 中的机器人恢复到启动位姿并打印校验状态，再读取轨迹、运行 KC-WBC
并发送控制指令，支持
`controller_mode:=none|per_link|kc`。
这两个 KC-WBC launch 默认显式使用 `ros_domain_id:=176`，隔离其他会话在
默认 domain 0 上发布的控制指令、时钟和状态。需要修改时，两端均传入相同的
`ros_domain_id:=<编号>`；launch 参数优先于终端的 `ROS_DOMAIN_ID`。
另开终端使用 `ros2 topic`、RViz 或自定义控制程序时，先设置 `ROS_DOMAIN_ID=176`
（或两端选用的编号）。仅关闭本 launch 的 adapter 并不能屏蔽同 domain 的其他发布者。
仅启动场景时，机器人在首个可见仿真帧之前恢复完整出生位姿、清零速度并进入
PhysX 休眠，保持静止待机；收到 `/isaac_joint_commands` 的驱动目标后由 PhysX
正常唤醒。这个初始化只执行一次，不在运行中反复锁定机器人位姿，外部碰撞仍可
唤醒机器人。出生变换应用于整个 URDF 容器，保证底盘、轮子和机械臂的初始关节
连接位置一致。
Stage 3 尚未通过严格验收：实际动力学约束和异步 ROS 跟踪仍有未通过项。
两个 launch 使用同一个 `stage_view:=<scene.usda>` 和出生位姿。轨迹跟随 launch
默认以 `mesh_source:=groundtruth` 首次从 USD 提取局部三角网格及保守 surface
voxels，之后复用 `<scene>.kc_mesh.npz`；也可用 `mesh_cache_file` 指定缓存。
`mesh_source:=lidar` 已接入前后两路实时点云；默认启用双 LiDAR，按时间戳配准、
过滤自身回波并更新 KC-SVSDF 表面体素地图。启动命令、限制和三轨迹对照见
[实时双 LiDAR 避障](LIDAR_WBC.md)。
