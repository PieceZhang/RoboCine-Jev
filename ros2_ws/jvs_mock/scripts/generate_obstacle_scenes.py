#!/usr/bin/env python3
"""Generate legacy checks plus three graded household-obstacle scenes."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]

legacy = [
 ('moveout','shot_moveout.json',[0]*6,
  [dict(name='base_bollard',type='sphere',center=[-.32,.17,.13],radius=.075),
   dict(name='arm_obstacle',type='sphere',center=[-.20,-.075,.72],radius=.055)]),
 ('combo','shot_combo1.json',[1.2,-.2,.4,0,-.2,.37],
  [dict(name='base_bollard',type='sphere',center=[.24,.24,.13],radius=.08),
   dict(name='arm_obstacle',type='sphere',center=[.25,-.085,.72],radius=.055)]),
]

graded = [
 ('household_easy', 'easy', [
  dict(name='side_table_leg',type='cylinder',center=[-.22,.38,.25],radius=.07,height=.50),
  dict(name='storage_cube',type='cube',center=[-.55,-.39,.09],size=.18),
  dict(name='door_stop_pyramid',type='triangular_pyramid',center=[-.82,.48,.15],size=[.18,.18,.30],yaw_deg=20),
 ]),
 ('household_medium', 'medium', [
  dict(name='plant_pot',type='cylinder',center=[-.24,.32,.21],radius=.075,height=.42),
  dict(name='footstool',type='cube',center=[-.55,-.38,.09],size=.18),
  dict(name='door_stop',type='triangular_pyramid',center=[-.90,.42,.13],size=[.16,.16,.26],yaw_deg=-15),
  dict(name='floor_lamp_base',type='cylinder',center=[-.78,-1.00,.28],radius=.08,height=.56),
  dict(name='side_storage',type='cube',center=[-.96,1.15,.10],size=.20),
 ]),
 ('household_hard', 'hard', [
  dict(name='tall_plant_pot',type='cylinder',center=[-.20,.32,.27],radius=.06,height=.54),
  dict(name='front_footstool',type='cube',center=[-.50,-.34,.07],size=.14),
  dict(name='front_door_stop',type='triangular_pyramid',center=[-.90,.34,.12],size=[.14,.14,.24],yaw_deg=25),
  dict(name='lamp_base',type='cylinder',center=[-1.25,-.34,.27],radius=.065,height=.54),
  dict(name='rear_storage_cube',type='cube',center=[-1.42,.35,.08],size=.16),
  dict(name='rear_door_stop',type='triangular_pyramid',center=[-1.72,-.36,.13],size=[.15,.15,.26],yaw_deg=-20),
  dict(name='left_table_leg',type='cylinder',center=[-.48,1.15,.31],radius=.075,height=.62),
  dict(name='right_storage_cube',type='cube',center=[-.80,-1.15,.09],size=.18),
 ]),
]

scenes = legacy + [
    (name, 'shot_moveout.json', [0]*6, obstacles)
    for name, _, obstacles in graded
]
levels = {name: level for name, level, _ in graded}
for name,traj,q,obstacles in scenes:
 data=dict(format='jvs_obstacles/v1',minimum_clearance_m=.025,trajectory=traj,initial_q=q,
           obstacles=obstacles)
 if name in levels:
  data['difficulty']=levels[name]
  data['layout_note']='All primitives stand on z=0; the open sides preserve a feasible bypass route.'
 (root/'scenes'/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
