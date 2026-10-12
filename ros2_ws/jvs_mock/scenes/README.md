# Graded household obstacle scenes

The three `household_*.json` scenes use the same 18 s MoveOut shot and initial
joint state, so their avoidance results are directly comparable.

| File | Level | Objects | Intended challenge |
|---|---:|---:|---|
| `household_easy.json` | Easy | 3 | Clear nominal center line |
| `household_medium.json` | Medium | 5 | One shallow detour through staggered furniture |
| `household_hard.json` | Hard | 8 | Dense alternating obstacles with open side bypasses |

Every cylinder, cube, and triangular pyramid has its bottom face on `z=0`.
Isaac renders the requested primitive while KC-SVSDF uses a conservative union
of sphere proxies. The planar guide includes the chassis radius, 25 mm required
clearance, and an additional 15 mm routing reserve.

Run one scene after building the ROS package:

```bash
ros2 launch jvs_mock jvs_kc_wbc_isaac.launch.py \
  trajectory_file:=$PWD/ros2_ws/jvs_mock/trajectories/shot_moveout.json \
  obstacle_file:=$PWD/ros2_ws/jvs_mock/scenes/household_medium.json
```

Regenerate all JSON files with:

```bash
python3 ros2_ws/jvs_mock/scripts/generate_obstacle_scenes.py
```
