# ros2_ws — cinema_traj → ROS 2 bridge

把 cinema_traj 导出的相机轨迹发布到 ROS 2（Humble / Jazzy 通用）：

- **主接口**：`nav_msgs/msg/Path` → `/cam_desired_traj`（**transient_local
  latched** + reliable —— 晚加入的订阅者也能拿到整条路径）
- 调试：`visualization_msgs/msg/MarkerArray` → `~/path_markers`
  （相机路径 LINE_STRIP + 起终点球 + 可选假人胶囊，仅供 RViz）
- 服务：`std_srvs/srv/SetBool` → `~/republish`（`data: true` 重发路径；
  `data: false` 停止周期重发）

轨迹文件用 **重定时后** 的导出（Y-up 通用格式，`traj_loader` 负责转到
ROS REP-103 风格 map 系：x=north、y=east、z=up，四元数 body→world、
roll=0、body +x = 视线方向 —— 映射被 `tests/test_traj_loader.py` 钉死）。

## 构建 & 运行（Linux ROS2 机器）

```bash
# 在本目录（ros2_ws/）下
colcon build --packages-select cinema_traj_bridge
source install/setup.bash

# 用重定时后的轨迹（推荐：满足机体能力约束的可执行轨迹）
ros2 run cinema_traj_bridge traj_server --ros-args \
    -p traj_file:=/abs/path/to/cinema_traj_out/<scene>/exports/traj_samples_retimed.json \
    -p frame_id:=map

# 可选：叠加假人胶囊（需要 scene JSON 里有 subject_track）
ros2 run cinema_traj_bridge traj_server --ros-args \
    -p traj_file:=... -p scene_file:=/abs/path/to/scene.json
```

参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `traj_file` | （必填） | `traj_samples.json` / `*_retimed.json` / `traj_samples.csv` |
| `scene_file` | `""` | 可选 scene JSON（读 `subject_track` 画假人） |
| `frame_id` | `map` | Path / Marker 的 frame |
| `start_offset_s` | `0.0` | t=0 对应的绝对时刻偏移（默认发布时刻） |
| `republish_period` | `0.0` | >0 时周期重发（调试用） |
| `path_step` | `1` | 降采样（重定时流 20Hz 很密，RViz 可用 5–20） |

## 验证

```bash
ros2 topic echo /cam_desired_traj --once | head -40     # 整条 latched 路径
ros2 topic hz /cam_desired_traj                          # latched 只发一次属正常
ros2 service call /cinema_traj_bridge/republish std_srvs/srv/SetBool '{data: true}'
rviz2   # Fixed Frame=map，Add → MarkerArray → /cinema_traj_bridge/path_markers
        #         Add → Path → /cam_desired_traj
```

## 本地（macOS / 无 ROS）验证

`cinema_traj_bridge/traj_loader.py` 是纯逻辑（无 rclpy），在仓库根目录：

```bash
python -m pytest tests/test_traj_loader.py tests/test_gz_player.py -q
python -m compileall ros2_ws/cinema_traj_bridge           # traj_server 语法检查
```

---

# Ubuntu 22.04 + ROS 2 Humble + Gazebo Fortress (gz-sim6) 测试

`cinema_traj_gz_demo` 包：在 **Gazebo Fortress / gz-sim6** 里复刻
`make_demo_room_pcd.py` 的合成房间（6×6×2.7m + 三盒子 + 开顶便于观察，地面/
光源内联、零 Fuel 依赖），内置 10Hz 降采样的重定时 demo 轨迹。`path_player`
通过 world 的 `/world/demo_room/set_pose` 服务（由 gz-sim **UserCommands**
系统提供，launch 里用 `ros_gz_bridge parameter_bridge` 桥成 ROS 2 的
`ros_gz_interfaces/srv/SetEntityPose`）把静态 `camera_drone` 实体沿轨迹搬运
（**运动学回放**，与我们的可执行轨迹语义一致；物理级/离板控制测试走
PX4 SITL 路线，见末节）。

## ① 环境安装（一次性）

```bash
# ROS 2 Humble Desktop（官方 docs.ros.org/humble/Installation/Ubuntu-Install-Debs）
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
     -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update && sudo apt install ros-humble-desktop

# Gazebo Fortress (gz-sim6) + ros_gz 集成（Humble 官方配对就是 Fortress，
# gz-fortress 会作为 ros-humble-ros-gz 的依赖一并装上）
sudo apt install ros-humble-ros-gz python3-colcon-common-extensions

echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && source ~/.bashrc
gz sim --version     # 应显示 Gazebo Sim, version 6.x（gz-sim6 / Fortress）
```

## ② 取代码并构建

```bash
git clone https://github.com/civicW/RoboCine.git
cd RoboCine/ros2_ws
colcon build --packages-select cinema_traj_bridge cinema_traj_gz_demo
source install/setup.bash
```

（demo 轨迹已随包安装：`trajectories/demo_room_retimed.json`。要测自己机器上
生成的轨迹，把 `cinema_traj_out/<scene>/exports/traj_samples_retimed.json`
scp 过来，launch 时传 `traj_file:=/abs/path.json`。）

## ③ 一键跑

```bash
ros2 launch cinema_traj_gz_demo demo_room.launch.py
#   默认 time_scale=10（重定时 demo 全长 244.8s，10× 约 25s 看完）
ros2 launch cinema_traj_gz_demo demo_room.launch.py time_scale:=25.0 loop:=true
ros2 launch cinema_traj_gz_demo demo_room.launch.py \
    traj_file:=/home/you/traj_samples_retimed.json
```

launch 会同时起四样东西：gz-sim6 加载 `demo_room.world`（`-r` 非暂停）、
`parameter_bridge`（把 `/world/demo_room/set_pose` 桥到 ROS 2）、
`traj_server`（latched `/cam_desired_traj` + RViz markers）、`path_player`。

Gazebo 里应看到：橙色四旋翼从起始位形起飞，绕黄色目标盒一圈（慢速拐角 =
能力约束的体现），随后跟拍路径走到房间另一角。

## ③′ 真实三输入 demo（Replica frl_apartment_0 + 假人 + 前推运镜）

`scripts/make_gz_world.py` 从精修场景 JSON 生成 `worlds/frl_apartment_0.world`
（内联墙面/楼梯/矮柜/扶手椅/茶几/电视墙/吊灯 + `camera_drone` + 假人 `subject_01`，
开顶便于观察，零 Fuel 依赖）。假人不用 `<actor>`（那需要外部 DAE 皮肤），
而是复用 `path_player` 的 set_pose 传送：launch 起**两个** player 实例，
一个搬相机、一个按同 schema 的 `frl_apartment_0_subject.json` 搬假人。

```bash
ros2 launch cinema_traj_gz_demo frl_apartment_0.launch.py
#   假人先走 0.3s 到 (0.5, -2.6) 停 10s；相机在她停住期间低机位仰角推近
#   （注视点全程锁定假人头部，重定时后镜头 2.0s，落在停顿窗口内）
ros2 launch cinema_traj_gz_demo frl_apartment_0.launch.py time_scale:=5.0 loop:=true
```

在 macOS 上重新生成 world / 轨迹文件（场景或轨迹更新后）：

```bash
python scripts/make_gz_world.py --scene cinema_traj/scenes/frl_apartment_0.json \
    --traj cinema_traj_out/frl_apartment_0/exports/traj_samples_retimed.json
```

两个 player 各自首个 timer tick 起表，launch 下基本同秒启动；主体轨迹
自带 10s 停顿缓冲，秒级启动偏差视觉上无感。

## ④ 验证清单

```bash
gz service -l | grep set_pose                        # gz 侧：/world/demo_room/set_pose
ros2 service list | grep set_pose                    # ROS 侧（bridge 后）
ros2 topic echo /cam_desired_traj --once | head -30  # latched 整条路径
ros2 service call /cinema_traj_bridge/republish std_srvs/srv/SetBool '{data: true}'
rviz2    # Fixed Frame:=map → Add: Path(/cam_desired_traj) + MarkerArray(~/path_markers)
```

也可以不走 ROS 直接手推一下实体验证 set_pose 链路：

```bash
gz service -s /world/demo_room/set_pose --reqtype gz.msgs.Pose \
    --reptype gz.msgs.Boolean --timeout 300 --req 'name: "camera_drone", position: {z: 1.8}'
```

## 故障排查

- **`path_player` 报 set_pose 不可用**：确认 `parameter_bridge` 节点在跑、
  world 名字是 `demo_room`（服务名 `/world/demo_room/set_pose` 二者必须一致）。
- **旧版 ros_gz 桥服务失败**（ros_gz [#881]/[#711]）：`sudo apt update &&
  sudo apt upgrade ros-humble-ros-gz-bridge ros-humble-ros-gz-interfaces`，
  Humble 的 ros_gz 仍在维护并持续回修。
- **ros_gz 与本机 Gazebo 版本错配**（如装过 Garden/Harmonic）：Humble 二进制
  对应 Fortress，确保 `gz sim --version` 是 6.x；多版本共存时设
  `export GZ_VERSION=fortress`。
- **gz-sim 无界面/服务器卡住**：`GZ_SIM_RESOURCE_PATH` 不需要（world 自包含）；
  远程无显示时可加 `gz_args:="-s -r <world>"` 只起 server。

[#881]: https://github.com/gazebosim/ros_gz/issues/881
[#711]: https://github.com/gazebosim/ros_gz/issues/711

## 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `traj_file` | 包内置 demo | 任意 exports JSON（retimed 优先） |
| `time_scale` | `10.0` | 回放加速（不影响轨迹本身的约束合规性） |
| `loop` | `false` | 循环回放 |
| `world` | `demo_room` | world 名（决定 set_pose 服务名） |
| `entity` | `camera_drone` | 被搬运的 gz 实体名（换模型改 world） |
| `rate_hz` | `30.0` | set_pose 调用频率 |

## 换成自己的轨迹 / 自己的房间

- 轨迹：`--traj_file` 传重定时 JSON 即可（坐标系自动映射 Y-up → gz z-up）
- 房间：world 是纯 SDF（gz-sim6 语法，系统插件 Physics/UserCommands/
  SceneBroadcaster 必须保留），照 `demo_room.world` 的墙/盒子条目改；或在
  macOS 端 `run_traj.py --stage extract --map your_map.pcd` 后照着 extract
  出的 OBB 手写对应盒子

## 下一步：物理级 / 离板控制测试（PX4 SITL 路线）

`path_player` 的 `set_pose` 回放是**运动学**的——足以验证轨迹几何、朝向与
能力约束的观感，但不涉及动力学与控制闭环。要做物理级测试（PX4 v1.14+
同样配对 Fortress）：

1. PX4 SITL + Gazebo Fortress（`PX4_GZ_WORLD` 选场景，`make px4_sitl gz_x500`）；
2. 写一个 Path → `px4_msgs/TrajectorySetpoint` 的转换节点（按轨迹时间轴
   逐点下发 position + yaw，重定时流 20Hz 正好匹配 offboard ≥2Hz 要求）；
3. offboard 模式起飞后切轨迹源，观察跟踪误差（拐角处应与 retime 预言的
   减速一致）。
