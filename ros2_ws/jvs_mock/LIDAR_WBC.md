# KC-WBC 实时双 LiDAR 避障

在仓库根目录构建并加载工作区后，分别运行：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 launch jvs_mock jvs_kc_wbc_isaac.launch.py \
  stage_view:=/home/w84384651/Documents/RoboCine-Yuelin/cinema_traj_out/kujiale_0021_kid_combo/simview/simview.usda\
  lidar_debug_view:=true
```

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 launch jvs_mock jvs_kc_wbc_trajfollow.launch.py \
  mesh_source:=lidar \
  trajectory_file:=$PWD/ros2_ws/jvs_mock/trajectories/shot_combo1.json \
  output_directory:=/tmp/jvs_combo_lidar
```

两端 `ros_domain_id` 必须一致，默认 176。Isaac 默认开启前后雷达，后雷达
`JVS_Nearfield` 发布 `/lidar/points`，前雷达 `Example_Rotary` 发布
`/lidar/front/points`。可用 `lidar_topic` / `front_lidar_topic` 在两端一致地改名。
一键启动脚本也支持 `scripts/run_kc_wbc_ros.sh --mesh-source lidar --headless`。

## 数据流

- 只从两路 ROS `PointCloud2` 获取障碍数据。默认发布逐帧扫描（`lidar_full_scan:=false`），按消息时间戳查询传感器到 `odom`
  的 TF，匹配机械臂状态，过滤无效量程、地面、高度范围外点和机器人自身回波。
- 密集点云处理在独立进程完成；反馈、点云与控制分别调度，避免回调排队。
  新体素要求两次间隔至少 30 ms、相隔不足 1 秒的观测确认，以过滤孤立噪点。
  使用默认 8 cm 表面体素，
  体素外接球半径为 `sqrt(3) * voxel_size / 2`，接入现有 KC-SVSDF、执行 CBF
  和连续命令 sweep。此接口不要求生成闭合三角网格。
- 执行层接收最新不可变地图；异步预览使用对应快照，每 2 秒重建平面路线。
  路线仍以轨迹初始状态为锚点，避免在线重建时移动相对轨迹的目标。
  转角处的 A* 恢复搜索也在预览进程执行；实时控制使用可见路径点，等待
  新路径时保持位置参考，所有实际指令仍通过最新地图的 CBF/sweep 检查。
  在线地图中，如果末段 MoveOut 需要绕过家具，前一镜头的停顿只准备
  朝向，避免地图投影更新把 MoveOut 平移提前消耗在切换阶段。
- 默认保留最近 30 秒、机器人周围 3 米的观测。遮挡没有被当作自由空间证据；
  地图也不使用 USD 真值补洞。定位依赖仿真提供的 odom/TF，不是 SLAM。
- 启动先丢弃 1 秒扫描缓冲，然后要求两台雷达均有有效观测。点云或状态超时
  会停止；点云默认允许最多 2 秒仿真时间延迟，另有 5 秒墙钟失联保护。
  启动缺失观测会输出失败报告，不会用空地图开始运动。
- Isaac TF 中 `base_link` 的父节点统一为 `odom`。源场景的摄影机预览模型、
  动画视锥线和轨迹辅助线隐藏，避免它们被 RTX 当作物理障碍；真值提取也
  排除这些辅助对象，缓存 fingerprint 会随此规则更新。

`mesh_voxel_size`、`mesh_map_radius`、`mesh_minimum_height`、`mesh_maximum_height`
和 `minimum_clearance` 同时适用于两种来源。`mesh_cache_file` 仅供 groundtruth
使用。LiDAR 输出目录包含 `lidar_map.npz`、`lidar_map.json`、最近扫描和
`telemetry.csv` 中的 `lidar_revision`、`lidar_voxels`、`lidar_age_s`、
`lidar_update_ms`。

地图是可见表面的近似；逐帧配准降低整圈累积的姿态误差，但消息没有逐点
时间字段，因此不是逐点运动补偿。
自身过滤有 6 cm 噪声带，传感器壳体也被过滤。后雷达输入使用 0.16 m
最小可信量程（标称 0.10 m 外加 0.06 m 边缘噪声带），前雷达为 1 m。未经观测的遮挡物、传感器盲区、
快速动态物体和过期观测仍限制避障能力。

## 三轨迹对比

```bash
python3 ros2_ws/jvs_mock/scripts/test_lidar_wbc_suite.py \
  --output /tmp/jvs_lidar_comparison --domain 224 --timeout 900 \
  --reference-mesh /tmp/jvs_lidar_comparison/reference_mesh.npz
```

脚本顺序运行 `shot_orbit`、`shot_moveout`、`shot_combo1` 的两种来源，每次拥有
独立启动的仿真、控制器、日志和结果文件。目录必须是新的，以免误读旧结果。
结束后 `compare_lidar_wbc.py` 使用同一真值地图复核两种方法的实际执行状态。
默认比较门槛：两者均无运行故障、通过跟踪精度/镜头运动/指令连续性检查，
所有记录状态满足 2.5 cm 真值间隙；LiDAR
位置/姿态 RMS 增量分别不超过 3 cm / 0.03 rad，执行时长不超过基线 1.25 倍；
逐镜头完成比例相差不超过 0.25，两者均通过运动方向检查。
该审计是控制采样时刻的几何检查，不代替每个物理步的接触或执行器约束验收。
报告中的 `comparable` 表示完成、避障间隙、指令连续性以及相对 RMS/时长
门槛通过；`passed` 还要求两种来源均通过原有的绝对跟踪和镜头运动验收。
脚本仅在全部门槛通过时返回 0，基线自身未达标也会保留非零退出状态。

## 实测结果

以下为重编号前旧 `shot_combo` 的历史结果，不能作为当前 `shot_combo1` 的 LiDAR
验收记录；当前编号/groundtruth 精度筛选见 [KC-WBC 文档](STAGE3_KC_WBC.md)。

三条输入轨迹的 LiDAR / groundtruth 对照均完成且无控制故障，相对性能、逐镜头完成比例/方向及真值采样间隙门槛通过。

| 轨迹 | 位置 RMS GT / LiDAR (cm) | LiDAR 真值最小间隙 (cm) |
|---|---:|---:|
| shot_orbit | 4.51 / 4.47 | 13.17 |
| shot_moveout | 2.38 / 2.28 | 13.31 |
| shot_combo | 23.88 / 21.70 | 3.62 |
