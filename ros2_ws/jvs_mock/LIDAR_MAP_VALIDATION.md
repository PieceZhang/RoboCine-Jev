# Kujiale 单 LiDAR 场地覆盖验证（2026-09-22）

## 结论

**当前 LiDAR 能获得有效局部点云，但本次验证没有获得完整场地 map。**
默认出生点无法看到隔墙房间与家具背面；9 个静态视点累积明显改善覆盖，但仍不完整。
这不等于“单 LiDAR 永远不能建图”：完成可达区域的移动探索、定位、地图融合后可能获得实用地图，
本次没有实现或验证这些能力。不能把未观测空间当成自由空间。

本次保留原有唯一传感器 `/jvs_mock/lidar_link/JvsRtxLidar`，
没有添加第二个 LiDAR，也没有继续修改安装位置或扫描参数：

- 相对 `base_link`：`xyz = (-0.085, 0, 0.235) m`，水平安装。
- `JVS_Nearfield`：近距 0.1 m、远距 30 m、64 线、水平 360°、垂直 ±45°、标称 10 Hz。
- 修复了该分支的 `jvs_simview_isaac.build_parser()` 引用未导入 `jvs` 导致的 `NameError`。
  CLI 回归测试直接启动 `--help`，无需启动 Isaac。

## 实测结果

参考物为输入场景 `/World/Environment` 的可见 mesh：1,107 个 mesh、2,583,723 个三角形，
按面积确定性采样 1,500,000 个点。统计采用 8 cm 表面体素、12 cm 匹配容差。
“障碍高度”是 Isaac 世界坐标 `z = 0.035–1.60 m`；原始场景为 Y-up，转换为 `(x, -z, y)`。

| 测试 | 原始场景、物理保留、单点 | 静态环境、单点 | 静态环境、9 视点 |
| --- | ---: | ---: | ---: |
| 实际 ROS 点云帧数 | 10 | 10 | 27（每点 3 帧） |
| 全场障碍表面覆盖 | 22.55% | 22.50% | 56.81% |
| 全场 0–3 m 高度表面覆盖 | 23.41% | 23.37% | 56.90% |
| 出生点 3 m 内障碍表面覆盖 | 46.26% | 46.04% | 83.48% |
| 出生点 1 m 内障碍表面覆盖 | 53.05% | 54.27% | 80.49% |
| 3 m 内低矮障碍表面覆盖（0.035–0.30 m） | 41.35% | 41.07% | 78.22% |
| 二维障碍表面投影覆盖 | 31.82% | 31.82% | 69.90% |

表面覆盖的分母包括遮挡面、家具背面等不可见表面，**不是传感器漏检率，也不是自由空间覆盖率**。
二维投影同样只是障碍表面的 XY 投影，不是已验证的 occupancy grid。
12 cm 容差可能让相邻遮挡面也被计为覆盖；JSON 同时提供 8/12/20 cm 容差结果。

为区分不可见与漏扫，另用 CPU 三角形射线求理想可见性：仅考虑静态环境、不透明材料、连续 ±45°视场；
不含机器人自遮挡、Cast、Props 和动画 CamBody，也不模拟离散线束。
静态单点的理论可见障碍表面占全场约 16.94%，其中实测匹配率 **94.72%**；
9 视点理论可见占约 47.87%，实测匹配率 **98.18%**。
这是辅助诊断，不是 RTX 材料/反射模型的真值或完整率承诺。

原始单点与静态单点非常接近，支持“场地遮挡是重要限制”的判断。
9 视点图仍可见床/柜/沙发背面等未观测区域，不能宣称已完整重建所有表面。

### 数据质量与限制

- 每帧包含 230,400 个有限 XYZ 点，`frame_id = lidar_link`。分析另外做量程过滤和机器人各 link 的
  15 mm 膨胀包围盒自过滤；这是审计用过滤，未接入生产点云链路，不是精确 mesh 自过滤。
- 原始单点固定外参最大变化为 0；多视点最大矩阵元素变化约 `5.54e-7`。
  ROS TF 验证为 `base_link -> lidar_link`，平移约 `(-0.085, 0, 0.235)`，旋转约单位四元数。
- 静态单点每帧在全部 72 个 5°方位桶中有环境候选回波。这不代表所有方位/高度无盲区；
  实际回波俯仰范围约 `[-29.286°, 45°]`，不等于配置中的全部负俯仰线束都能看到环境。
- 静态单点相邻帧障碍体素 Jaccard 为 0.839–0.871，多视点内为 0.861–0.948。
  原场景保留 Cast、Props、CamBody、Annotations 及作者设置的动画，而参考分母仅含 Environment。
  因此观测 PLY 可以含人物、道具、相机代理等额外回波，不可直接作为纯静态建图成果。
- 原始单点观测体素到参考采样面的距离中位数约 1.67 cm，P95 约 56.1 cm。
  后者包含参考高度范围外与 Environment 之外的回波，不能解释为测距误差；也未将这些不匹配点隐瞒或全部删除。
- 接收的连续扫描时间戳相隔约 0.25 s（仿真时钟）。本次受限资源的审计运行**不构成标称 10 Hz 的吞吐验收**。
- 点云世界配准使用模拟器真值矩阵，不是 ROS 定位/SLAM；静态多视点通过重设位姿采样，
  未验证视点间路线可达、碰撞安全、运动去畸变或动态地图清理。
- 原始场景物理测试虽然成功采集，但日志有 9 条嵌套刚体层级错误，以及资产 `displayColor/displayOpacity`
  长度不一致警告。没有修改这些共享资产，也不能据此宣称原场景物理通过验收。
- 早期冷缓存尝试被本任务超时/定点停止，没有形成有效扫描；最终结果只取 `CAPTURE_COMPLETE` 的三次运行。

## 隔离、输入与产物

- 基于原有 `codex-lidar-nearfield-single` 的 `f40016c`；最终修改直接提交该分支，不新建结果分支。
- 工作目录：`/tmp/robocine_lidar_map_muSTNb`。主工作区源码、工作分支、共享场景与其他测试进程没有被修改或停止。
- 本次 `ROS_DOMAIN_ID=221`，loopback-only Fast DDS UDP，关闭 builtin/shared-memory transports。
  ROS graph context 明确使用同一 domain，不使用 ROS daemon。复现时必须重新检查 domain 是否空闲。
- 每次仅运行一个本任务 Isaac 实例；Kit `--portable-root`、ROS 日志、XDG/CUDA 缓存及 colcon 输出均在临时 worktree。
  本任务顺序复用自己的 Kit 缓存；限制 8 个 Kit CPU 线程、64×64 viewport，关闭 Motion BVH。
  仍与其他工作共享 GPU/CPU 硬件，不声称完全没有硬件资源竞争。
- 输入：`/home/w84384651/Documents/RoboCine-Yuelin/cinema_traj_out/kujiale_0021_kid_combo/simview/simview.usda`。
- 输入顶层文件 SHA-256：`d740ff0169ca963f44707d21d6028301e93973bcd80b8852f76f8a46a3b3338c`。
  这是顶层文件哈希，不是所有引用资产的联合哈希；引用层清单见 `reference/reference.json`。

结果根目录为本文件同目录的 `results/kujiale_map_validation/`（生成数据被 Git 忽略，保留在临时 worktree）：

- `original_physics/capture.json`、`scan_*.npz`：原始场景 10 帧及 TF。
- `single_pose_retry/`、`multi_pose/`：静态环境 10 帧、9 视点 27 帧。
- [原始单点覆盖图](results/kujiale_map_validation/original_analysis/coverage.png)、
  [9 视点覆盖图](results/kujiale_map_validation/multi_analysis/coverage.png)。灰色为参考中尚未观测的表面。
- 每个 `*_analysis/` 包含 `metrics.json`、`coverage.png`、`observed.png`、`observed_world.ply`、
  `unobserved_reference.ply`、`map.npz`；`visibility.json` 是理想可见性诊断。
  `unobserved_reference.ply` 是真值参考，绝不能并入观测地图假装已扫描。
- `static_scene.usda` 是引用原始场景的临时 override，只移除 207 个 prim 上的环境 physics/physx API，
  保留视觉几何、材质与作者动画；机器人和额外地面仍保留物理。

## 复现

在独立临时 worktree 中执行，先检查空闲 domain、GPU 余量和其他测试；不要直接在共享主工作区运行。
需要 ROS Humble、Isaac Sim 5.1 的 Python（含 USD/Warp）、系统 Python 的 NumPy/SciPy/Matplotlib。
原始场景及其相对引用资产应保持完整。

```bash
source /opt/ros/humble/setup.bash
cd /tmp/robocine_lidar_map_muSTNb/ros2_ws
colcon build --packages-select jvs_mock --symlink-install
cd ..
export ROS_DOMAIN_ID=221  # 仅本次已检查；复现时重新选择空闲值
export ISAAC_PYTHON=/home/w84384651/miniconda3/envs/env_isaaclab/bin/python
export OMNI_KIT_ACCEPT_EULA=yes
SCENE=/home/w84384651/Documents/RoboCine-Yuelin/cinema_traj_out/kujiale_0021_kid_combo/simview/simview.usda
PKG=$PWD/ros2_ws/jvs_mock
OUT=$PKG/results/new_lidar_audit

# 原始场景物理集成：output 必须是新目录，避免覆盖旧扫描。
bash "$PKG/scripts/run_lidar_map_capture.sh" --stage "$SCENE" \
  --keep-environment-physics --output "$OUT/original" --scans 10
```

离线参考与静态层不启动 Kit/ROS/GPU。以下路径匹配本次安装；其他机器应选择其安装对应的 USD 库目录。

```bash
ISAAC_SITE=/home/w84384651/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages
USD_LIB=$ISAAC_SITE/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
export OPENBLAS_NUM_THREADS=2 PXR_WORK_THREAD_LIMIT=2
PYTHONPATH="$USD_LIB" LD_LIBRARY_PATH="$USD_LIB/bin:/home/w84384651/miniconda3/envs/env_isaaclab/lib" \
  "$ISAAC_PYTHON" "$PKG/scripts/lidar_map_reference.py" --stage "$SCENE" --output "$OUT/reference"
PYTHONPATH="$USD_LIB" LD_LIBRARY_PATH="$USD_LIB/bin:/home/w84384651/miniconda3/envs/env_isaaclab/lib" \
  "$ISAAC_PYTHON" "$PKG/scripts/lidar_static_stage.py" --stage "$SCENE" --output "$OUT/static_scene.usda"

# 确认前一个实例完全退出后，才能复用它的 Kit portable root。
bash "$PKG/scripts/run_lidar_map_capture.sh" --stage "$OUT/static_scene.usda" \
  --output "$OUT/multi" --kit-root "$OUT/original/kit" --scans 3 \
  --poses '[[5,1,0],[3.4,2.5,0],[7.8,0.5,0],[10,0,0],[12.4,1.5,0],[13,3.5,0],[1.8,2.5,0],[3.2,-2,0],[7.7,-1.5,0]]'

/usr/bin/python3 "$PKG/scripts/analyze_lidar_map.py" --reference "$OUT/reference/reference.npz" \
  --capture "$OUT/multi" --output "$OUT/multi_analysis"
PYTHONPATH="$ISAAC_SITE/isaacsim/extscache/omni.warp.core-1.8.2+lx64" \
  "$ISAAC_PYTHON" "$PKG/scripts/lidar_visibility_reference.py" \
  --reference "$OUT/reference/reference.npz" --map "$OUT/multi_analysis/map.npz" \
  --output "$OUT/multi_analysis/visibility.json"

PYTHONPATH=ros2_ws/jvs_mock /usr/bin/python3 -m pytest -q ros2_ws/jvs_mock/test
```

本次 CPU 回归：**21 passed**；新增覆盖坐标转换、体素化、自过滤、空采集拒绝、未知表面保留及 simview CLI。
新增脚本通过 `py_compile`，启动脚本通过 `bash -n`。Isaac 三次有效采集均正常退出。

后续若要验收“完整可用 map”，应单独定义可达区域/高度/分辨率/未知比例标准，再验证单 LiDAR 的移动探索、
定位融合与自过滤/动态物体清理。仅修改安装位置或增大扫描角不能消除隔墙遮挡。
