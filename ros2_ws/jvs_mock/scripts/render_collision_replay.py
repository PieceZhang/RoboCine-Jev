#!/usr/bin/env python3
"""MP4 replay of Isaac telemetry (scientific reconstruction, not rendered video)."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,FFMpegWriter
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jvs_mock.kc_svsdf import CollisionGeometry
from jvs_mock.jvs_kinematics import forward_kinematics

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();rows=list(csv.DictReader((a.run/'telemetry.csv').open()));scene=json.loads(a.scene.read_text())
get=lambda key:np.array([float(r[key]) for r in rows])
t=get('time_s');states=np.column_stack([get('state_'+k) for k in ['x','y','yaw']+[f'q{i}' for i in range(1,7)]])
actual=np.column_stack([get('actual_camera_'+k) for k in 'xyz']);target=np.column_stack([get('target_camera_'+k) for k in 'xyz'])
quat=np.column_stack([get('actual_camera_q'+k) for k in 'xyzw']);error=get('position_error')*1000;clearance=get('clearance_m')*1000
model=CollisionGeometry();fig=plt.figure(figsize=(12,7));grid=fig.add_gridspec(2,2,width_ratios=[1.4,1])
ax=fig.add_subplot(grid[:,0],projection='3d');e=fig.add_subplot(grid[0,1]);c=fig.add_subplot(grid[1,1])
for o in scene['obstacles']:
 u,v=np.meshgrid(np.linspace(0,2*np.pi,24),np.linspace(0,np.pi,12));x,y,z=o['center'];r=o['radius']
 ax.plot_surface(x+r*np.cos(u)*np.sin(v),y+r*np.sin(u)*np.sin(v),z+r*np.cos(v),color='tomato',alpha=.5)
ax.plot(*target.T,color='green',ls='--',alpha=.45,label='camera target')
ax.plot(*actual.T,color='orange',alpha=.25,label='actual camera path')
arm,=ax.plot([],[],[],color='#1977b8',lw=5,marker='o',ms=3,label='robot kinematic chain')
base,=ax.plot([],[],[],color='#1977b8',lw=2)
cam,=ax.plot([],[],[],color='orange',marker='o',ms=7)
camera_axes=[ax.plot([],[],[],color=color,lw=2)[0] for color in ['red','green','blue']]
goal,=ax.plot([],[],[],color='green',marker='x',ms=7)
xy=np.r_[states[:,:2],np.array([o['center'][:2] for o in scene['obstacles']])]
center=(xy.min(axis=0)+xy.max(axis=0))/2;span=max(np.ptp(xy,axis=0).max()+.6,1.1)
ax.set_xlim(center[0]-span/2,center[0]+span/2);ax.set_ylim(center[1]-span/2,center[1]+span/2);ax.set_zlim(0,1.2)
ax.set_box_aspect([span,span,1.2]);ax.set_xlabel('X [m]');ax.set_ylabel('Y [m]');ax.set_zlabel('Z [m]');ax.view_init(25,-62);ax.legend(fontsize=7)
e.plot(t,error,color='#1977b8');e.set_ylabel('Camera position error [mm]');e.grid(alpha=.2)
c.plot(t,clearance,color='#1977b8');c.axhline(scene['minimum_clearance_m']*1000,color='red',ls='--',label='required clearance');c.set_ylabel('Measured clearance [mm]');c.set_xlabel('Time [s]');c.grid(alpha=.2);c.legend(fontsize=8)
mark_e=e.axvline(0,color='black');mark_c=c.axvline(0,color='black');title=fig.suptitle('')
for pane in [e,c]:
 if 'combo' in a.run.name:
  for junction in [22,36]:pane.axvline(junction,color='gray',ls=':')
indices=np.unique(np.r_[np.arange(0,len(t),10),len(t)-1])
def update(frame):
 i=indices[frame];x=states[i];cp,cr=forward_kinematics(x[3:]);br=Rotation.from_quat(quat[i]).as_matrix()@cr.T
 z=actual[i,2]-(br@cp)[2];frames,_,_=model.frames(x,z,br)
 points=np.array([frames[k][0] for k in range(8)])
 arm.set_data_3d(*points.T)
 theta=np.linspace(0,2*np.pi,48);local=np.c_[.18*np.cos(theta),.18*np.sin(theta),np.full(len(theta),.07)]
 points=local@br.T+np.r_[x[:2],z];base.set_data_3d(*points.T)
 cam.set_data_3d([actual[i,0]],[actual[i,1]],[actual[i,2]]);goal.set_data_3d([target[i,0]],[target[i,1]],[target[i,2]])
 camera_rotation=Rotation.from_quat(quat[i]).as_matrix()
 for j,line in enumerate(camera_axes):
  line.set_data_3d(*np.array([actual[i],actual[i]+.07*camera_rotation[:,j]]).T)
 mark_e.set_xdata([t[i],t[i]]);mark_c.set_xdata([t[i],t[i]])
 title.set_text(f"Isaac telemetry replay | {rows[i]['collision_mode']} | {rows[i]['shot_type']} | t={t[i]:.1f}s\n"+
                f"v={float(rows[i]['command_v']):.3f} m/s, omega={float(rows[i]['command_omega']):.3f} rad/s | 3x playback")
 return arm,base,cam,goal,mark_e,mark_c,title
fig.tight_layout(rect=[0,0,1,.91]);a.output.parent.mkdir(parents=True,exist_ok=True)
anim=FuncAnimation(fig,update,frames=len(indices),interval=1000/15,blit=False)
anim.save(str(a.output),writer=FFMpegWriter(fps=15,bitrate=1800));plt.close(fig)
print(a.output)
