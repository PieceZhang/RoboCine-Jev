#!/usr/bin/env python3
"""scene JSON (+subject track) → Gazebo Fortress (gz-sim6) world SDF +
playback trajectory files for cinema_traj_gz_demo.

Axis convention (pinned by demo_room.world): gz is z-up with
gz (x, y, z) = cinema_traj world (x, z, y) — x = +X (north), y = +Z (east),
z = +Y (up). Yaw about gz +z uses the same atan2(dz, dx) as the world
heading, so heading values carry over unchanged.

The mannequin reuses the camera's set_pose teleport mechanism (path_player)
instead of an <actor>: actors need an external DAE skin, and the demo runs
fully offline. Outputs:

    worlds/<scene>.world            static room + furniture + camera_drone
                                     + subject mannequin (open top: ceiling
                                     omitted for GUI visibility — collision
                                     truth stays in the scene JSON)
    trajectories/<scene>_retimed.json   copy of the retimed camera samples
    trajectories/<scene>_subject.json   mannequin poses in traj_samples schema

    python scripts/make_gz_world.py --scene cinema_traj/scenes/frl_apartment_0.json \
        --traj cinema_traj_out/frl_apartment_0/exports/traj_samples_retimed.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vlm_intention.common.atomic_io import atomic_write_json

# label keyword → SDF material (ambient, diffuse)
_MATERIALS = [
    (("沙发", "beanbag", "椅", "chair", "坐", "床", "bed"), ("0.36 0.30 0.55", "0.45 0.38 0.68")),
    (("桌", "table", "茶几"), ("0.52 0.36 0.18", "0.62 0.44 0.24")),
    (("书架", "shelf", "柜", "cabinet"), ("0.38 0.30 0.22", "0.48 0.38 0.28")),
    (("灯", "lamp", "light"), ("0.95 0.85 0.35", "1.0 0.92 0.5")),
    (("毯", "rug", "地毯"), ("0.75 0.62 0.25", "0.85 0.72 0.35")),
    (("楼梯", "stair"), ("0.55 0.57 0.60", "0.65 0.67 0.70")),
]


def _material_for(label: str):
    for keys, mat in _MATERIALS:
        if any(k.lower() in label.lower() for k in keys):
            return mat
    return ("0.30 0.55 0.40", "0.38 0.65 0.48")            # default furniture green


def _rpy_from_matrix(R) -> tuple:
    """Rotation matrix → SDF (roll, pitch, yaw) — R = Rz(yaw)·Ry(pitch)·Rx(roll)."""
    yaw = math.atan2(R[1][0], R[0][0])
    pitch = math.atan2(-R[2][0], math.hypot(R[0][0], R[1][0]))
    roll = math.atan2(R[2][1], R[2][2])
    return roll, pitch, yaw


def _world_to_gz_pose(center, rotation, half_height_shift: float = 0.0) -> str:
    """OBB center(+rotation, world y-up) → gz <pose> string (x y z r p y).

    gz (x, y, z) = world (x, z, y); the rotation permutes the same way:
    R_gz = P·R·Pᵀ with P the (x,y,z)→(x,z,y) swap.
    """
    x, y, z = center[0], center[2], center[1] + half_height_shift
    P = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    Rg = [[sum(P[i][k] * rotation[k][j] for k in range(3)) for j in range(3)]
          for i in range(3)]
    Rg = [[sum(Rg[i][k] * P[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
    roll, pitch, yaw = _rpy_from_matrix(Rg)
    return (f"{x:.4f} {y:.4f} {z:.4f} "
            f"{roll:.4f} {pitch:.4f} {yaw:.4f}")


def _box_model(name: str, obb: dict, ambient: str, diffuse: str,
               collision: bool = True, opacity: float = 1.0) -> str:
    ext = obb["extents"]
    pose = _world_to_gz_pose(obb["center"], obb.get("rotation") or
                             [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    col = (f'<collision name="col"><geometry><box><size>'
           f'{ext[0]:.4f} {ext[2]:.4f} {ext[1]:.4f}</size></box></geometry></collision>'
           if collision else "")
    return f"""    <model name="{name}"><static>true</static><pose>{pose}</pose>
      <link name="link">
        {col}
        <visual name="vis">
          <geometry><box><size>{ext[0]:.4f} {ext[2]:.4f} {ext[1]:.4f}</size></box></geometry>
          <material>
            <ambient>{ambient} {opacity}</ambient><diffuse>{diffuse} {opacity}</diffuse>
          </material>
        </visual>
      </link>
    </model>"""


def _mannequin_model(subject_id: str, height_m: float) -> str:
    """Simple capsule mannequin facing +x, feet at z=0 (teleported by a
    second path_player instance)."""
    r = 0.18
    body_h = 0.74 * height_m
    head_z = 0.92 * height_m
    return f"""    <model name="{subject_id}"><static>true</static><pose>0 0 0 0 0 0</pose>
      <link name="link">
        <visual name="body"><pose>0 0 {body_h / 2:.3f} 0 0 0</pose>
          <geometry><cylinder><radius>{r}</radius><length>{body_h:.3f}</length></cylinder></geometry>
          <material><ambient>0.85 0.55 0.60 1</ambient><diffuse>0.9 0.62 0.68 1</diffuse></material>
        </visual>
        <visual name="head"><pose>0 0 {head_z:.3f} 0 0 0</pose>
          <geometry><sphere><radius>0.11</radius></sphere></geometry>
          <material><ambient>0.95 0.80 0.70 1</ambient><diffuse>0.98 0.85 0.75 1</diffuse></material>
        </visual>
      </link>
    </model>"""


# imported verbatim from demo_room.world (offline, no model:// includes)
_CAMERA_DRONE = """    <model name="camera_drone"><static>true</static><pose>0 0 1.5 0 0 0</pose>
      <link name="body">
        <visual name="chassis">
          <geometry><box><size>0.30 0.30 0.08</size></box></geometry>
          <material><ambient>1.0 0.45 0.05 1</ambient><diffuse>1.0 0.45 0.05 1</diffuse></material>
        </visual>
        <visual name="rotor_0"><pose>0.11 0.11 0.07 0 0 0</pose>
          <geometry><cylinder><radius>0.08</radius><length>0.02</length></cylinder></geometry>
          <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material>
        </visual>
        <visual name="rotor_1"><pose>-0.11 0.11 0.07 0 0 0</pose>
          <geometry><cylinder><radius>0.08</radius><length>0.02</length></cylinder></geometry>
          <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material>
        </visual>
        <visual name="rotor_2"><pose>0.11 -0.11 0.07 0 0 0</pose>
          <geometry><cylinder><radius>0.08</radius><length>0.02</length></cylinder></geometry>
          <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material>
        </visual>
        <visual name="rotor_3"><pose>-0.11 -0.11 0.07 0 0 0</pose>
          <geometry><cylinder><radius>0.08</radius><length>0.02</length></cylinder></geometry>
          <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material>
        </visual>
        <visual name="camera"><pose>0.14 0 0 0 0 0</pose>
          <geometry><box><size>0.08 0.08 0.06</size></box></geometry>
          <material><ambient>0.05 0.05 0.05 1</ambient><diffuse>0.05 0.05 0.05 1</diffuse></material>
        </visual>
        <collision name="col"><geometry><box><size>0.30 0.30 0.08</size></box></geometry></collision>
      </link>
    </model>"""


def build_world_sdf(scene: dict, world_name: str) -> str:
    room = scene["rooms"][0]
    lo, hi = room["bounds_min"], room["bounds_max"]
    cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    sx, sz = hi[0] - lo[0], hi[2] - lo[2]
    height = hi[1] - lo[1]

    models = []
    # floor patch on the ground plane (visual only; ground plane collides)
    models.append(_box_model("room_floor",
                             {"center": [cx, lo[1] + 0.01, cz], "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              "extents": [sx, 0.02, sz]},
                             "0.62 0.50 0.38", "0.72 0.59 0.45", collision=False))
    for st in scene["structures"]:
        if st["kind"] == "ceiling":
            continue                        # open top for GUI visibility
        if st["kind"] == "floor":
            continue                        # ground plane + floor patch above
        mat = ("0.80 0.80 0.82", "0.85 0.85 0.87") if st["kind"] == "wall" else \
              ("0.55 0.57 0.60", "0.65 0.67 0.70")
        models.append(_box_model(st["id"], st["obb"], *mat))
    for ob in scene["objects"]:
        ambient, diffuse = _material_for(ob["label"])
        collide = ob["obb"]["extents"][1] > 0.10           # rugs stay visual-only
        models.append(_box_model(ob["id"], ob["obb"], ambient, diffuse, collision=collide))
    models.append(_CAMERA_DRONE)
    if scene.get("subject_track"):
        models.append(_mannequin_model(scene["subject_track"]["subject_id"],
                                       scene["subject_track"].get("height_m", 1.75)))

    return f"""<?xml version="1.0" ?>
<!-- {world_name}.world — AUTO-GENERATED by scripts/make_gz_world.py from
{scene.get('provenance', {}).get('source_map', 'scene JSON')}. Gazebo Fortress
(gz-sim6), z-up: gz (x, y, z) = cinema_traj (x, z, y). Open top (ceiling in the
scene JSON is collision truth; omitted here for visibility). The mannequin and
camera_drone are static entities teleported via /world/{world_name}/set_pose by
two path_player instances. Fully offline: no Fuel / model:// dependencies. -->
<sdf version="1.7">
  <world name="{world_name}">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>{cx:.2f} {cz:.2f} {height + 8:.2f} 0 0 0</pose>
      <diffuse>0.85 0.85 0.85 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <light type="point" name="room_fill">
      <pose>{cx:.2f} {cz:.2f} {height * 0.8:.2f} 0 0 0</pose>
      <diffuse>0.35 0.35 0.38 1</diffuse>
      <attenuation><range>15</range></attenuation>
    </light>

    <model name="ground_plane"><static>true</static>
      <link name="link">
        <collision name="col">
          <geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>
        </collision>
        <visual name="vis">
          <geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>
          <material><ambient>0.45 0.5 0.45 1</ambient><diffuse>0.45 0.5 0.45 1</diffuse></material>
        </visual>
      </link>
    </model>

{chr(10).join(models)}
  </world>
</sdf>
"""


def subject_track_to_samples(track: dict, step_s: float = 0.2) -> dict:
    """subject_track frames → traj_samples.json schema (feet on floor, yaw =
    heading, pitch 0) so path_player can teleport the mannequin."""
    frames = sorted(track["frames"], key=lambda f: f["t"])
    samples = []
    for f in frames:
        if samples and f["t"] - samples[-1]["t"] < step_s - 1e-6:
            continue
        samples.append({"t": round(f["t"], 3), "x": f["pos"][0], "y": f["pos"][1],
                        "z": f["pos"][2], "yaw_deg": round(math.degrees(f["heading"]), 2),
                        "pitch_deg": 0.0, "roll_deg": 0.0, "segment": "subject"})
    if samples[-1]["t"] != frames[-1]["t"]:
        f = frames[-1]
        samples.append({"t": round(f["t"], 3), "x": f["pos"][0], "y": f["pos"][1],
                        "z": f["pos"][2], "yaw_deg": round(math.degrees(f["heading"]), 2),
                        "pitch_deg": 0.0, "roll_deg": 0.0, "segment": "subject"})
    return {"scene_id": track.get("subject_id", "subject"),
            "frame": {"up": "+Y", "units": "meters", "yaw_zero": "+Z",
                      "note": "mannequin feet track (path_player teleport source)"},
            "count": len(samples), "samples": samples}


def main() -> None:
    ap = argparse.ArgumentParser(description="scene JSON → gz world + 播放轨迹文件")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--traj", help="retimed traj_samples_retimed.json（复制进包）")
    ap.add_argument("--out-pkg", default="ros2_ws/cinema_traj_gz_demo")
    a = ap.parse_args()

    scene = json.loads(Path(a.scene).read_text(encoding="utf-8"))
    scene_id = scene["scene_id"]
    pkg = _REPO_ROOT / a.out_pkg
    worlds, trajs = pkg / "worlds", pkg / "trajectories"
    worlds.mkdir(parents=True, exist_ok=True), trajs.mkdir(parents=True, exist_ok=True)

    (worlds / f"{scene_id}.world").write_text(build_world_sdf(scene, scene_id),
                                              encoding="utf-8")
    print(f"✅ world → {worlds / (scene_id + '.world')}")

    if scene.get("subject_track"):
        out = trajs / f"{scene_id}_subject.json"
        atomic_write_json(out, subject_track_to_samples(scene["subject_track"]))
        print(f"🚶 subject track → {out}（{json.loads(out.read_text())['count']} 采样点）")

    if a.traj:
        out = trajs / f"{scene_id}_retimed.json"
        out.write_text(Path(a.traj).read_text(encoding="utf-8"))
        print(f"📹 retimed traj → {out}")


if __name__ == "__main__":
    main()
