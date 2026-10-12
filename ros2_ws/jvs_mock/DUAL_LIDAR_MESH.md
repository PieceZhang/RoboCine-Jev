# 双 LiDAR 场景 mesh

本次修改在 `yuelin-lidar-nearfield-single` 分支上继续，不修改共享主 worktree。
用户最新需求允许两个 LiDAR，取代该分支之前的“仅一个 LiDAR”限制。
旧报告 `LIDAR_MAP_VALIDATION.md` 是单雷达历史基线，不代表当前配置。

## 安装与接口

| 传感器 | base_link 下 xyz / rpy | Isaac 配置 | ROS 点云 / frame |
| --- | --- | --- | --- |
| 保留后雷达 | `(-0.085,0,0.235)` / `(0,0,0)` | `JVS_Nearfield` | `/lidar/points` / `lidar_link` |
| 恢复原始前雷达 | `(0.190,0,0.274)` / `(0,0.898844565,0)` | `Example_Rotary` | `/lidar/front/points` / `front_lidar_link` |

前雷达位置、俯仰角、Isaac 配置名和 Gazebo 扫描参数恢复自提交 `a743ee6`。
为兼容现有后雷达消费者，原来的后雷达 TF 和话题不改名；前雷达使用新名称。
后雷达配置文件未修改。两台传感器各有 render product 和 publisher，共享 ROS context/仿真时钟。
默认同时启动；`enable_lidar:=false` 全部关闭，`enable_front_lidar:=false` 只关闭前雷达。
现在也可用 `lidar_mode:=front` 只开启前雷达，或 `lidar_mode:=both`（默认）同时开启；
此模式适用于 Isaac、simview 和 Gazebo。旧的 `enable_front_lidar:=false` 保留仅后雷达行为。
前雷达有独立的 `front_lidar_config`/`front_lidar_topic` 参数，不能与后雷达共用话题。

模式切换回归：在同一临时 worktree 中，使用独立 ROS domain `223`
和顺序复用的私有 Isaac Kit 缓存，对下述静态视觉场景层分别实际采集一帧：

| 模式 | stage 中 RTX LiDAR 数 | ROS 点云 | 结果目录 |
| --- | ---: | --- | --- |
| `front` | 1 | `/lidar/front/points`，`front_lidar_link`，207,092 点 | `results/lidar_mode_validation/front_warm/` |
| 默认 `both` | 2 | `/lidar/points`，`lidar_link`，230,400 点；前雷达同上，206,851 点 | `results/lidar_mode_validation/both_default/` |

两次 Isaac 采集正常完成；Gazebo Xacro 展开分别含 1 / 2 个传感器，
Gazebo launch 声明 `lidar_mode` 的 `front` / `both` 选择及 `both` 默认值。
CPU 回归 30 项通过，worktree 内 colcon 构建通过。场景源文件仍是下文指定的 USDA；
本次运行使用先前生成的静态视觉层，未重新评估场地覆盖率或重建 mesh。
首次使用全新冷缓存的尝试在尚未采集时停止，统计只使用上述完成的两次运行。

前雷达 `Example_Rotary` 原始名义量程 1–200 m，恢复它不等于增加另一个近场雷达。
Gazebo 前雷达恢复原始 96 线、0–60° 垂直扫描；这与 Isaac profile 并非相同物理扫描模型。
本次 Gazebo 只做配置/展开回归，不宣称进行了 Gazebo GPU 实测。

## 点云融合与重建方式

1. `capture_lidar_map.py` 订阅两路真实 ROS `PointCloud2`，核对各自 frame，
   确认 stage 恰好有两台 LiDAR，每个视点每台收集 3 个完整扫描。
   两路通过共同预热条件后才记录，避免较快的前雷达提前完成采集。
2. 扫描保留传感器名、仿真时间、传感器到 world/base 的变换及机器人各 link 视觉包围盒。
   去掉非有限值、量程外点及落在机器人包围盒（膨胀 15 mm）内的自反射点。
   每个传感器分别检查外参稳定性，不把两套不同外参误报成漂移。
3. `reconstruct_lidar_mesh.py` 不读取源 USD 或参考模型，只读扫描和记录的外参。
   每台雷达/每视点取首个扫描，按 0.4° 角度网格保留真实最近回波，在角度平面三角化；
   删除边长超过 0.25 m、角度间隔超过 3°、退化或极细的三角形。
   统一到 Isaac Z-up world 米制坐标，按 3 cm 网格焊接顶点，再去重/删除退化面和过长边。
4. 导出真正含三角面的 `scene_mesh.ply` 和 `scene_mesh.obj`，以及 `mesh.npz`、`mesh_report.json`。
   `analyze_lidar_map.py` 单独使用场景参考表面评价覆盖率；参考表面从不用于重建或补洞。

这是**观测表面的离线、局部三角 mesh**，不是完整密闭场景或可直接验收的碰撞网格。
视点通过仿真静态重定位得到，不是自主探索/导航；配准用仿真世界真值，不是 SLAM。
外参在接收扫描时读取，且 Motion BVH 关闭，不能把这一采集流程直接用于运动机器人建图。
两路发布周期不同，共同预热不等于逐点/硬件同步；动态人物仍可能产生残影。
多视点三角片可能重叠或有非流形边；报告明确记录边界/非流形边，未假装闭合修补成功。
最大边长只限制局部插值跨度，不保证所有小空隙都被保留，不能作为安全避障唯一依据。

## 隔离与复现

### 2026-09-22 实测结果

最终结果位于 `results/dual_lidar_validation/`：

- `original_sync/`：原始 USD 保留物理，两台雷达各 3 帧，共 6 帧。
- `multi_sync/`：9 个静态视点，每台每视点 3 帧，共 54 帧。
- `mesh/scene_mesh.obj` / `mesh/scene_mesh.ply`：九视点融合 mesh，
  **203,783 顶点、459,428 三角面**；使用 18 帧（每台/每视点首帧）。
- `original_sync_mesh/`：原场景单点 mesh，34,123 顶点、60,440 三角面。
- `analysis_both/`、`analysis_rear/`、`analysis_front/`：同一次九视点采集的对照分析，
  含点云 `observed_world.ply`、覆盖图 `coverage.png` 和 `metrics.json`。
- `original_sync_analysis/`：保留原始物理的单点分析。

覆盖率沿用单雷达报告口径：8 cm 参考表面体素，距实测回波不超过 12 cm，
包含不可见背面/遮挡面，不是自由空间网格或完整性验收。**下表是点云覆盖率，不是三角面覆盖率。**

| 九视点同场对照 | 仅后雷达 | 仅前雷达 | 双雷达 |
| --- | ---: | ---: | ---: |
| 全场障碍表面，高 0.035–1.60 m | 56.79% | 28.96% | **58.11%** |
| 全场表面，高 0–3 m | 56.90% | 31.53% | **59.36%** |
| 初始点周围 3 m 障碍表面 | 83.48% | 53.61% | **85.30%** |
| 初始点周围 3 m 低障碍，高 0.035–0.30 m | 77.94% | 50.54% | **80.66%** |
| XY 障碍表面投影 | 70.05% | 45.38% | **71.63%** |

双雷达全场障碍表面覆盖相对后雷达增加 **1.32 个百分点**，不是获得完整场地。
原始物理场景单点双雷达的全场障碍覆盖为 23.81%，高 0–3 m 表面覆盖 25.46%。
多视点双雷达合计 81,726 个 8 cm 实测点云体素。

两台雷达每个视点首帧时间差最大 50 ms；前雷达连续时间戳间隔约 0.10 s，
后雷达约 0.25 s。不是标称吞吐量验收，也没有逐点运动补偿。
九视点的每传感器外参最大元素变化 `4.22e-7`，原始物理单点 `5.65e-7`。
ROS TF 实测后雷达位置约 `(-.085,0,.235)`、单位旋转；
前雷达位置约 `(.190,0,.274)`、四元数 xyzw 约 `(0,.434445,0,.900698)`。

九视点 mesh 有 67,465 条边界边、37,549 条非流形边；面积统计 760.73 m²
包含重叠片，不能当作“已恢复真实面积”。它适合查看/继续处理，不是 watertight 或碰撞认证模型。
点云相对参考表面的中位距离 1.54 cm、95 分位 47.14 cm；参考仅包含 Environment，
不含额外人物/道具/相机代理等，因此该残差不是单纯量程误差，也没有删除全部离群点来美化覆盖。

原始 USD 仍出现此前已记录的 9 条嵌套 RigidBody/xformstack 错误及材质 primvar 警告；
采集正常退出不代表原场景物理结构通过验收。静态多视点 run 没有 `[Error]` 日志。
本轮早期 `original/`、`multi/` 为共同预热修正前的试运行，最终统计不用这些数据。

CPU 回归 **29 passed**，覆盖双传感器配置/独立发布器、外参分别统计、真实回波三角化、
跨传感器坐标融合、孔洞跨度限制、退化/重复面清理以及已有控制器测试。
独立 worktree 内 `colcon build --packages-select jvs_mock --symlink-install` 成功；
修改脚本通过 `py_compile`、`git diff --check`。本轮 Isaac 测试全部正常退出，
最终检查无 domain 222 UDP 端口残留，其他两个 GPU 进程仍在运行。

### 运行说明

工作目录：`/tmp/robocine_lidar_map_muSTNb`，所有 build/install/log/results 均在此目录。
本轮检查后使用 `ROS_DOMAIN_ID=222`，FastDDS UDP 仅 loopback、禁用共享内存传输。
Kit 使用本 worktree 内上轮测试的私有 portable root，严格顺序复用；未改全局 Isaac 安装和缓存。
不切换主 worktree 分支，不停止其他进程，不改源 USD。共用物理 GPU 仍有资源竞争，
因此本测试限制线程/渲染尺寸；不宣称拥有独占硬件。

源场景：`/home/w84384651/Documents/RoboCine-Yuelin/cinema_traj_out/kujiale_0021_kid_combo/simview/simview.usda`。
顶层文件 SHA-256：`d740ff0169ca963f44707d21d6028301e93973bcd80b8852f76f8a46a3b3338c`。
单点集成直接加载此原始场景并保留物理；九视点用上一轮只移除物理 API 的隔离 sublayer，
仍引用同一场景视觉几何与动画。没有修改源场景文件。

以下从本 worktree 根目录执行；再次运行时重新检查 domain 是否空闲，选择新的输出目录。

```bash
source /opt/ros/humble/setup.bash
(cd ros2_ws && colcon build --packages-select jvs_mock --symlink-install)
export ROS_DOMAIN_ID=222
export ISAAC_PYTHON=/home/w84384651/miniconda3/envs/env_isaaclab/bin/python
export OMNI_KIT_ACCEPT_EULA=yes OPENBLAS_NUM_THREADS=2
PKG=$PWD/ros2_ws/jvs_mock
OUT=$PKG/results/new_dual_audit
SCENE=/home/w84384651/Documents/RoboCine-Yuelin/cinema_traj_out/kujiale_0021_kid_combo/simview/simview.usda
bash "$PKG/scripts/run_lidar_map_capture.sh" --stage "$SCENE" \
  --keep-environment-physics --scans 3 --output "$OUT/original"

# 静态 sublayer/reference 的生成方法见 LIDAR_MAP_VALIDATION.md。
# 多视点仅是可见性诊断：这些离散位置没有被验证为可达、无碰撞的导航路径。
bash "$PKG/scripts/run_lidar_map_capture.sh" --stage "$OUT/static_scene.usda" \
  --kit-root "$OUT/original/kit" --output "$OUT/multi" --scans 3 \
  --poses '[[5,1,0],[3.4,2.5,0],[7.8,0.5,0],[10,0,0],[12.4,1.5,0],[13,3.5,0],[1.8,2.5,0],[3.2,-2,0],[7.7,-1.5,0]]'
/usr/bin/python3 "$PKG/scripts/reconstruct_lidar_mesh.py" \
  --capture "$OUT/multi" --output "$OUT/mesh"
/usr/bin/python3 "$PKG/scripts/analyze_lidar_map.py" \
  --capture "$OUT/multi" --reference "$OUT/reference/reference.npz" --output "$OUT/analysis_both"
# 添加 --sensor rear 或 --sensor front，比较同一次采集中的单路结果。
PYTHONPATH=ros2_ws/jvs_mock /usr/bin/python3 -m pytest -q ros2_ws/jvs_mock/test
```

采集时 `--rear-only` 可用来单独启用后雷达回归；前雷达 link 仍在机器人模型中。
脚本拒绝覆盖已有扫描/mesh。生成的大文件保留在 `results`（git ignored），不纳入代码提交。
