#!/usr/bin/env python3
"""record_cam_error — record & plot measured-vs-desired camera pose error.

RECORD mode (run while the demo loop is alive; needs the Isaac backend for
/tf + /joint_states):

    ros2 run ... # no — plain python3, ROS Humble system python:
    python3 scripts/record_cam_error.py record --out /tmp/cam_err.npz

Subscribes /tf (world→base_link), /joint_states, /cmd_vel, /tracking_start
and the latched /cam_desired_traj; dumps everything with sim-clock stamps.

PLOT mode (offline, after the pass):

    python3 scripts/record_cam_error.py plot --npz /tmp/cam_err.npz \
        --png cinema_traj_out/kujiale_0021_kid_dolly/cam_tracking_error.png

Measured camera pose is rebuilt exactly like the tracking node: base pose
from TF (Isaac Z-up) converted to plan Y-up, arm FK from measured joints,
camera_link convention (look = +x, upright). t=0 of the pass = arrival of
the /tracking_start message (node enters TRACK within one control tick,
20 ms).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

_REPO = os.environ.get("ROBOCINE_REPO",
                       "/home/w84384651/Documents/RoboCine")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from cinema_traj.core.jarvis import camera_from_look, fk_camera  # noqa: E402


def _tf_to_matrix(tr, rot) -> np.ndarray:
    x, y, z, w = rot.x, rot.y, rot.z, rot.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tr.x, tr.y, tr.z]
    return T


# --------------------------------------------------------------------------- #
# record
# --------------------------------------------------------------------------- #
def _load_path_offline(path_json: str) -> list:
    """Rebuild the (t, T) path list from a traj_samples(_retimed).json.

    Mirrors _on_path's message-side conversion sample-for-sample (positions
    straight from the Y-up export; look from look_vector_ros mapped back to
    Y-up), so plots do not depend on which side produced the path.
    """
    from cinema_traj_bridge.traj_loader import load_samples, look_vector_ros

    poses = []
    for s in load_samples(path_json):
        v = look_vector_ros(s["yaw_deg"], s["pitch_deg"])
        look = np.array([v[0], v[2], v[1]])          # ROS map → Y-up world
        n = float(np.linalg.norm(look))
        look = look / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        yaw = math.atan2(look[0], look[2])
        pitch = math.asin(min(1.0, max(-1.0, look[1])))
        T = np.eye(4)
        T[:3, :3] = camera_from_look(yaw, pitch)
        T[:3, 3] = [s["x"], s["y"], s["z"]]
        poses.append((float(s["t"]), T))
    return poses


def record(out_path: str, duration_s: float | None,
           path_json: str | None) -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Path
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage

    class Recorder(Node):
        def __init__(self):
            super().__init__("cam_error_recorder",
                             automatically_declare_parameters_from_overrides=True)
            self.set_parameters([rclpy.parameter.Parameter(
                "use_sim_time", rclpy.Parameter.Type.BOOL, True)])
            latched = QoSProfile(depth=1,
                                 durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=QoSReliabilityPolicy.RELIABLE)
            self.tf_t, self.tf_base = [], []
            self.js_t, self.js_q = [], []
            self.cmd_t, self.cmd_vw = [], []
            self.trig_t = None
            self.path = None
            self.create_subscription(TFMessage, "/tf", self._on_tf, 50)
            self.create_subscription(JointState, "/joint_states",
                                     self._on_js, 50)
            self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 50)
            self.create_subscription(String, "/tracking_start", self._on_trig,
                                     10)
            # The latched 493-pose Path (~100 KB) republished every 10 s has
            # been observed to kill this participant's fastrtps link ~1 s
            # after the SECOND delivery; --path-json skips the subscription.
            if path_json is None:
                self.create_subscription(Path, "/cam_desired_traj",
                                         self._on_path, latched)
            else:
                self.path = _load_path_offline(path_json)
                self.get_logger().info(
                    f"path loaded offline: {len(self.path)} poses")
            self._t0 = None

        def _now(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        def _on_tf(self, msg: TFMessage) -> None:
            for t in msg.transforms:
                if (t.header.frame_id == "world"
                        and t.child_frame_id == "base_link"):
                    self.tf_t.append(self._now())
                    self.tf_base.append(_tf_to_matrix(t.transform.translation,
                                                      t.transform.rotation))

        def _on_js(self, msg: JointState) -> None:
            try:
                idx = [msg.name.index(f"joint{i}") for i in range(1, 7)]
                q = [float(msg.position[i]) for i in idx]
            except (ValueError, AttributeError, IndexError):
                return
            self.js_t.append(self._now())
            self.js_q.append(q)

        def _on_cmd(self, msg: Twist) -> None:
            self.cmd_t.append(self._now())
            self.cmd_vw.append([msg.linear.x, msg.angular.z])

        def _on_trig(self, msg: String) -> None:
            if self.trig_t is None:
                self.trig_t = self._now()
                self.get_logger().info(f"tracking_start seen at t={self.trig_t:.2f}")

        def _on_path(self, msg: Path) -> None:
            poses = []
            for ps in msg.poses:
                p, o = ps.pose.position, ps.pose.orientation
                pos = np.array([p.x, p.z, p.y])          # map → Y-up world
                qw, qx, qy, qz = o.w, o.x, o.y, o.z
                x_body = np.array([1 - 2 * (qy * qy + qz * qz),
                                   2 * (qx * qy + qw * qz),
                                   2 * (qx * qz - qw * qy)])
                look = np.array([x_body[0], x_body[2], x_body[1]])
                n = float(np.linalg.norm(look))
                look = look / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
                yaw = math.atan2(look[0], look[2])
                pitch = math.asin(min(1.0, max(-1.0, look[1])))
                # same rotation convention as the tracking node's path
                # rebuild (camera_from_look), NOT an ad-hoc yaw/pitch
                # product — the two differ by a fixed rotation
                T = np.eye(4)
                T[:3, :3] = camera_from_look(yaw, pitch)
                T[:3, 3] = pos
                t = ps.header.stamp.sec + ps.header.stamp.nanosec * 1e-9
                poses.append((t, T))
            self.path = poses

    rclpy.init()
    node = Recorder()
    import signal
    import time
    stopped = []

    def _save_and_stop(signum, _frame):
        # SIGTERM (timeout(1) default) must still dump the npz
        stopped.append(signum)

    signal.signal(signal.SIGTERM, _save_and_stop)
    signal.signal(signal.SIGINT, _save_and_stop)
    t_start = time.time()
    print(f"recording → {out_path}"
          + (f" for {duration_s:.0f}s" if duration_s else " — Ctrl-C to stop"),
          flush=True)
    try:
        while rclpy.ok() and not stopped:
            rclpy.spin_once(node, timeout_sec=0.1)
            if duration_s and (time.time() - t_start) > duration_s:
                break
    except KeyboardInterrupt:
        pass
    data = dict(
        tf_t=np.array(node.tf_t), tf_base=np.array(node.tf_base),
        js_t=np.array(node.js_t), js_q=np.array(node.js_q),
        cmd_t=np.array(node.cmd_t),
        cmd_vw=np.array(node.cmd_vw) if node.cmd_vw else np.zeros((0, 2)),
        trig_t=(np.array([node.trig_t]) if node.trig_t is not None
                else np.zeros(0)),
        path_t=np.array([p[0] for p in node.path]) if node.path else np.zeros(0),
        path_T=np.array([p[1] for p in node.path]) if node.path
               else np.zeros(0),
    )
    np.savez(out_path, **data)
    n_all = len(data["tf_t"]), len(data["js_t"]), len(data["cmd_t"])
    print(f"saved {out_path}: tf={n_all[0]} js={n_all[1]} cmd={n_all[2]} "
          f"path={len(data['path_t'])} trig={len(data['trig_t'])}", flush=True)
    node.destroy_node()
    rclpy.shutdown()


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #
def _rot_log(R: np.ndarray) -> np.ndarray:
    """so(3) log of a rotation matrix (axis*angle)."""
    c = (np.trace(R) - 1.0) / 2.0
    th = math.acos(min(1.0, max(-1.0, c)))
    if th < 1e-9:
        return np.zeros(3)
    if abs(math.pi - th) < 1e-6:          # near-π: use the symmetric part
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.maximum(np.diag(A), 1e-12))
        k = k * np.sign(A[np.argmax(k), :].sum()) if k.max() > 1e-9 else k
        return k / max(np.linalg.norm(k), 1e-12) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * math.sin(th)) * th


def _sample(path_t: np.ndarray, path_T: np.ndarray, t: float):
    """Linear position + slerp-ish rotation sample at t_rel (same as node)."""
    t = min(max(t, 0.0), path_t[-1])
    lo, hi = 0, len(path_t) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if path_t[mid] <= t:
            lo = mid
        else:
            hi = mid
    Ta, Tb = path_T[lo], path_T[hi]
    f = ((t - path_t[lo]) / max(path_t[hi] - path_t[lo], 1e-9))
    T = Ta.copy()
    T[:3, 3] = (1.0 - f) * Ta[:3, 3] + f * Tb[:3, 3]
    w = _rot_log(Ta[:3, :3].T @ Tb[:3, :3]) * f
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        R = np.eye(3)
    else:
        k = w / th
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]],
                      [-k[1], k[0], 0.0]])
        R = np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)
    T[:3, :3] = Ta[:3, :3] @ R
    return T, t >= path_t[-1] - 1e-9


def plot(npz_path: str, png_path: str, title: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz_path)
    if not len(d["trig_t"]):
        raise SystemExit("no /tracking_start recorded — pass never fired")
    if not len(d["path_t"]):
        raise SystemExit("no /cam_desired_traj recorded")
    t0 = float(d["trig_t"][0])
    # a trigger the tracker missed (ros2 topic pub --once discovery race)
    # must not define t=0: fall back to the first nonzero /cmd_vel when the
    # two disagree by more than 1 s
    nz_mask = (np.abs(d["cmd_vw"]).max(axis=1) > 1e-4) if len(d["cmd_t"]) \
        else np.zeros(0, dtype=bool)
    if nz_mask.any():
        t_cmd = float(d["cmd_t"][np.argmax(nz_mask)])
        print(f"first nonzero /cmd_vel at {t_cmd:.2f} (trigger {t0:.2f})")
        if abs(t_cmd - t0) > 0.3:
            print(f"trigger stamp is {t_cmd - t0:.1f}s off cmd_vel onset "
                  "(callback lag) — using cmd_vel onset as t=0")
            t0 = t_cmd
    # path relative times (same normalization as the tracking node)
    pt = d["path_t"] - d["path_t"][0]

    # measured camera pose (Y-up plan axes) per TF sample
    ts, e_p, e_o, p_m, p_d = [], [], [], [], []
    for t, T_base, in zip(d["tf_t"], d["tf_base"]):
        j = int(np.searchsorted(d["js_t"], t))
        j = min(j, len(d["js_t"]) - 1)
        q = d["js_q"][j]
        bx, bz = float(T_base[0, 3]), float(-T_base[1, 3])
        fwd = T_base[:3, 0]
        yaw = math.atan2(fwd[0], -fwd[1])
        T_cam = fk_camera(bx, bz, yaw, q)
        T_des, _ = _sample(pt, d["path_T"], t - t0)
        ts.append(t - t0)
        e_p.append(float(np.linalg.norm(T_des[:3, 3] - T_cam[:3, 3])))
        e_o.append(float(np.linalg.norm(
            _rot_log(T_des[:3, :3] @ T_cam[:3, :3].T))))
        p_m.append(T_cam[:3, 3])
        p_d.append(T_des[:3, 3])
    ts, e_p, e_o = np.array(ts), np.array(e_p), np.array(e_o)
    in_pass = (ts >= 0) & (ts <= pt[-1] + 0.5)
    post = ts > pt[-1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(title or "jarvis WBC tracking — camera pose error vs "
                 "/cam_desired_traj (kid dolly-out, retimed)")
    ax = axes[0, 0]
    ax.plot(ts[in_pass], e_p[in_pass] * 100, ".-", ms=2, lw=0.8, color="C0",
            label="during pass")
    ax.plot(ts[post], e_p[post] * 100, ".-", ms=2, lw=0.8, color="C1",
            alpha=0.5, label="hold @ end")
    ax.set_ylabel("position error [cm]")
    ax.set_xlabel("t since tracking_start [s]")
    ax.grid(alpha=0.3)
    ax.legend()
    m = e_p[in_pass]
    print(f"pass: mean {m.mean()*100:.1f} cm, max {m.max()*100:.1f} cm, "
          f"final(hold) {e_p[post][-5:].mean()*100 if post.any() else float('nan'):.1f} cm")

    ax = axes[1, 0]
    ax.plot(ts[in_pass], np.degrees(e_o[in_pass]), ".-", ms=2, lw=0.8,
            color="C0")
    ax.plot(ts[post], np.degrees(e_o[post]), ".-", ms=2, lw=0.8, color="C1",
            alpha=0.5)
    ax.set_ylabel("rotation error [deg]")
    ax.set_xlabel("t since tracking_start [s]")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    p_m, p_d = np.array(p_m), np.array(p_d)
    for k, c, lbl in zip(range(3), "rgb", "xyz"):
        ax.plot(ts[in_pass], p_m[in_pass, k] * 100, color=c, label=f"meas {lbl}")
        ax.plot(ts[in_pass], p_d[in_pass, k] * 100, color=c, ls="--", lw=0.8)
    ax.set_ylabel("camera position [cm]")
    ax.set_xlabel("t [s]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    if len(d["cmd_t"]):
        c = (d["cmd_t"] >= t0 - 0.5) & (d["cmd_t"] <= t0 + pt[-1] + 1.0)
        ax.plot(d["cmd_t"][c] - t0, d["cmd_vw"][c, 0], label="v_fwd [m/s]")
        ax.plot(d["cmd_t"][c] - t0, d["cmd_vw"][c, 1], label="omega [rad/s]")
        ax.legend(fontsize=7)
    ax.set_ylabel("base commands")
    ax.set_xlabel("t [s]")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    print(f"plot → {png_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--duration", type=float, default=None)
    r.add_argument("--path-json", default=None,
                   help="load the desired path offline from a traj_samples"
                        "(_retimed).json instead of subscribing the latched"
                        " /cam_desired_traj (whose 10 s republishes can"
                        " break the fastrtps link mid-recording)")
    p = sub.add_parser("plot")
    p.add_argument("--npz", required=True)
    p.add_argument("--png", required=True)
    p.add_argument("--title", default=None,
                   help="plot suptitle (default: kid dolly-out wording)")
    a = ap.parse_args()
    if a.mode == "record":
        record(a.out, a.duration, a.path_json)
    else:
        plot(a.npz, a.png, a.title)


if __name__ == "__main__":
    main()
