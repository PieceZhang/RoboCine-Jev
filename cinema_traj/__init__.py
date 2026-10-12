"""cinema_traj — From 3D map, Motion Prompt, Subject Pose, to Trajectory.

CinemaTraj-style (arXiv 2607.26910) camera/robot trajectory planner:

    3D map (PCD/PLY/scene JSON) ──▶ scene graph (rooms / OBB objects / doors)
    motion prompt (自然语言)     ──▶ LLM 分解为原子运镜序列 (1-3-1 pattern)
    subject pose track (可选)   ──▶ 动态主体追踪扩展 (track_follow / track_orbit / track_lead)
                                       │
               anchor 打分 + 参数化轨迹 + 两阶段 SDF 梯度优化
                                       │
                       轨迹 JSON + CSV + 离线 HTML 3D viewer

Paper-faithful core: 3-layer toolset (anchor selector / 13 object-level
movements / arc transition), anchor 5-term scoring, parametric trajectories
with (theta_fix, theta_free), two-pass differentiable SDF optimization.
Novel extension: time-parameterized dynamic subject as moving capsule
obstacle + occlusion target + subject-relative tracking primitives.
"""

__version__ = "0.1.0"
