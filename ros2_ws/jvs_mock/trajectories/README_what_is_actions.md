# 带 `_actions` 后缀的轨迹有什么区别？

以本目录中的 `shot_combo1.json` 和 `shot_combo1_actions.json` 为例：

- **不带 `_actions`**：**保存完整轨迹**，顶层 `samples` 使用整条轨迹的统一时间轴，包含动作、起始对焦、过渡和停留；`shots` 中已写好各动作的执行时间区间。
- **带 `_actions`**：保存分段动作序列（`jvs_relative_shot_sequence/v1`），**文件只记录动作本身，不预存过渡轨迹**。加载时由 `motion_sequence.py` 自动生成起始对焦、动作间的最小 jerk 平滑过渡及末尾停留。

