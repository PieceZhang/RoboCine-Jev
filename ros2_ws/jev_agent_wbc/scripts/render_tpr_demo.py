#!/usr/bin/env python3
"""Render the logged TPR demo with physical video or collision-model geometry.

No planner is rerun, no robot/camera pose is changed, and time is never paused.
"""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_agent_wbc.kc_svsdf import CollisionGeometry

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTS={s:ImageFont.truetype(FONT,s) for s in [18,20,22,24,28,32,40]}
BG='#0c1523';FG='#e8eff8';MUTED='#96a9c0';GREEN='#59deb2';RED='#ff977d';BLUE='#6db7ff';GOLD='#ffcf6e'


def text(d,xy,value,size=24,color=FG):
    d.text(xy,str(value),font=FONTS[size],fill=color)


def chart(d,box,times,series,colors,limits,title,current,labels=None):
    x,y,w,h=box; lo,hi=limits
    d.rounded_rectangle((x,y,x+w,y+h),radius=12,fill='#142236')
    text(d,(x+18,y+10),title,22)
    left,top,right,bottom=x+50,y+55,x+w-18,y+h-35
    for v in np.linspace(lo,hi,4):
        yy=bottom-(v-lo)/(hi-lo)*(bottom-top)
        d.line((left,yy,right,yy),fill='#2b3a50')
        text(d,(x+5,yy-10),f'{v:.2g}',18,MUTED)
    for values,color in zip(series,colors):
        points=[(left+t/times[-1]*(right-left),bottom-(v-lo)/(hi-lo)*(bottom-top))
                for t,v in zip(times,values) if t<=current]
        if len(points)>1:d.line(points,fill=color,width=3)
    xx=left+current/times[-1]*(right-left);d.line((xx,top,xx,bottom),fill=FG,width=1)
    text(d,(left,bottom+5),'0 s',18,MUTED);text(d,(right-55,bottom+5),f'{times[-1]:.0f} s',18,MUTED)
    if labels:
        for i,(label,color) in enumerate(zip(labels,colors)):
            text(d,(x+260+140*i,y+13),label,18,color)


def geometry_renderer():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(10.8,7.4),dpi=100,facecolor='#eff2f5')
    ax=fig.add_axes([0,0,1,1],projection='3d',facecolor='#eff2f5')
    geometry=CollisionGeometry()

    def mesh(kind,size):
        if kind=='box':
            v=np.array([[i,j,k] for i in (-1,1) for j in (-1,1) for k in (-1,1)])*size/2
            return [v[list(f)] for f in [(0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5)]]
        if kind=='sphere':
            theta=np.linspace(0,2*np.pi,20);phi=np.linspace(0,np.pi,12)
            v=np.array([[[size[0]*np.sin(p)*np.cos(t),size[0]*np.sin(p)*np.sin(t),size[0]*np.cos(p)] for t in theta] for p in phi])
            return [v[[i,i+1,i+1,i],[j,j,j+1,j+1]] for i in range(len(phi)-1) for j in range(len(theta)-1)]
        radius,top,height=(size[0],size[0],size[1]) if kind=='cylinder' else size
        theta=np.linspace(0,2*np.pi,20,endpoint=False)
        a=np.c_[radius*np.cos(theta),radius*np.sin(theta),np.full(20,-height/2)]
        b=np.c_[top*np.cos(theta),top*np.sin(theta),np.full(20,height/2)]
        return [a,b]+[np.array([a[i],a[(i+1)%20],b[(i+1)%20],b[i]]) for i in range(20)]
    locals_=[mesh(s[2],s[3]) for s in geometry.shapes]

    def render(row,reference,radius):
        ax.clear();ax.set_axis_off()
        ax.set(xlim=(-.4,.6),ylim=(-.45,.45),zlim=(-.02,1.02))
        ax.set_box_aspect((1.,.9,1.04));ax.view_init(elev=17,azim=-65)
        state=np.array(row['state']);br=np.array(row['base_rotation']) if 'base_rotation' in row else None
        frames=geometry.frames(state,row.get('base_z',0.),br)[0]
        for shape,polygons in zip(geometry.shapes,locals_):
            body,name,kind,size,offset,rot=shape;p,r=frames[body]
            color='#dab043' if name=='base_link' else '#2d8bd0' if 1<=body<=6 else '#303945' if 'wheel' in name or body==7 else '#97a5b0'
            world=[face@(r@rot).T+p+r@offset for face in polygons]
            ax.add_collection3d(Poly3DCollection(world,facecolors=color,edgecolors=color,linewidths=.3))
        center=np.array(row['obstacle'])
        ax.add_collection3d(Poly3DCollection([p+center for p in mesh('sphere',[radius])],facecolors='#ec6e36',edgecolors='#e76731',linewidths=.2))
        rp=np.array(reference)
        for axis,color in zip(np.eye(3),['#d86b64','#48a888','#498fcd']):
            points=np.array([rp,rp+axis*.12]);ax.plot(*points.T,color=color,linewidth=2)
        for v in np.arange(-.4,.61,.1):
            ax.plot([-.4,.6],[v,v],[0,0],color='#ccd3dc',linewidth=.6)
            ax.plot([v,v],[-.4,.4],[0,0],color='#ccd3dc',linewidth=.6)
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3])
    return render


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--fps',type=float,default=12.)
    args=parser.parse_args()
    output=args.output or args.run/'demo.mp4'
    rows=json.loads((args.run/'telemetry.json').read_text());summary=json.loads((args.run/'summary.json').read_text())
    times=np.array([r['t'] for r in rows]); duration=times[-1]
    physical=summary['backend'].startswith('PhysX')
    videos=(args.run/'video/overview.mp4').exists()
    metadata=json.loads((args.run/'video/frames.json').read_text()) if videos else []
    if physical and videos:
        from audit_tpr_video import audit
        if not audit(args.run)['pass_motion']:
            raise RuntimeError('visible obstacle motion does not match telemetry; see video_motion_audit.json')
    if physical and (args.run/'video/camera_output.json').exists():
        comparison=args.run/'video/comparison.mp4'
        if not comparison.is_file():
            raise RuntimeError('default three-camera PhysX video is missing')
        if comparison.resolve()!=output.resolve():
            shutil.copy2(comparison,output)
        print(output)
        return
    frame_times=np.array([r['t'] for r in metadata])
    renderer=None if videos else geometry_renderer()
    reference=summary.get('received_camera_position',rows[0]['camera_position'])
    pos=np.array([r['position_error_m'] for r in rows])*1000
    angle=np.rad2deg([r['orientation_error_rad'] for r in rows])
    distances=np.array([r['clearance_m'] for r in rows])*100
    frozen=np.array([r.get('frozen_clearance_m',np.nan) for r in rows])*100
    state=np.array([r['state'] for r in rows])
    tmp=output.with_suffix('.raw.mp4');tmp.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(1920,1080))
    if not writer.isOpened():raise RuntimeError('video encoder unavailable')
    sample_step=max(1,len(rows)//800);plot_t=times[::sample_step]
    for k,t in enumerate(np.arange(0,duration,1/args.fps)):
        idx=min(int(np.searchsorted(times,t)),len(rows)-1);row=rows[idx]
        image=Image.new('RGB',(1920,1080),BG);d=ImageDraw.Draw(image)
        text(d,(35,22),'TPR-WBC  |  相机固定，全身构型避障',40)
        subtitle='Isaac Sim / PhysX 实测状态' if physical else '加速度受限运动学闭环 · 无位姿投影'
        text(d,(38,80),subtitle+'   ·   障碍物连续飞行，WBC 不修改 received trajectory',22,MUTED)
        if videos:
            fi=min(int(np.searchsorted(frame_times,t)),len(frame_times)-1)
            overview=Image.open(args.run/f'video/overview/{fi:05d}.jpg')
            # A fixed crop only enlarges the same physical camera image.
            overview=overview.crop((120,125,840,650)).resize((1080,790))
        else:
            overview=renderer(row,reference,summary.get('obstacle_radius_m',.07)).resize((1080,790))
        image.paste(overview,(30,130));d=ImageDraw.Draw(image)
        calibration=args.run/'video/overview_calibration.json'
        if videos and calibration.exists():
            cal=json.loads(calibration.read_text())
            local=np.array(cal['rotation']).T@(np.array(reference)-np.array(cal['position']))
            focal=.5*cal['height']/np.tan(np.deg2rad(cal['vfov_deg']/2))
            px=30+(cal['width']/2-focal*local[1]/local[0]-120)*1080/720
            py=130+(cal['height']/2-focal*local[2]/local[0]-125)*790/525
            for sign in (-1,1):
                d.line((px+sign*28,py-20,px+sign*42,py-20),fill=GREEN,width=3)
                d.line((px+sign*42,py-20,px+sign*42,py+20),fill=GREEN,width=3)
                d.line((px+sign*42,py+20,px+sign*28,py+20),fill=GREEN,width=3)
            text(d,(px+55,py-20),'固定相机参考',20,'#16865d')
        text(d,(50,945),f'{t:05.1f} / {duration:.0f} s',32)
        status=row['info'].get('tpr_status','unknown')
        text(d,(365,950),{'normal':'相机任务保持','warning':'相机保持 / 构型余量预警','degraded':'安全降级：相机任务有残差'}.get(status,status),24,GREEN if status=='normal' else GOLD)
        text(d,(50,1005),'橙色：飞行障碍物    蓝色：机械臂    黄色：差速底盘',22,MUTED)
        if physical and not summary.get('physical_audit',{}).get('pass_constraints',False):
            text(d,(50,1042),'原始物理约束审计未通过；完整限制见运行报告。',20,GOLD)
        text(d,(1140,140),'固定相机任务 / 实际误差',28)
        text(d,(1140,187),f'位置  {row["position_error_m"]*1000:.4f} mm',32,GREEN)
        text(d,(1510,187),f'姿态  {np.rad2deg(row["orientation_error_rad"]):.4f}°',32,GREEN)
        text(d,(1140,242),f'全身最小间距  {row["clearance_m"]*100:.2f} cm',28,BLUE)
        text(d,(1140,285),'相机目标在整个演示中完全相同；所有曲线来自闭环日志。',20,MUTED)
        series=[distances[::sample_step]];colors=[BLUE];labels=['实际']
        if np.isfinite(frozen).all():series.append(frozen[::sample_step]);colors.append(RED);labels.append('机身不动')
        series.append(np.full(len(plot_t),2.5));colors.append(GOLD);labels.append('安全间距')
        chart(d,(1140,335,750,255),plot_t,series,colors,(-2.,max(25.,distances.max()*1.1)),
              '间距 / cm',t,labels)
        chart(d,(1140,610,750,240),plot_t,[pos[::sample_step]],[GREEN],(0.,max(.01,float(pos.max())*1.1)),
              '相机位置误差 / mm',t)
        chart(d,(1140,870,750,185),plot_t,[np.rad2deg(state[::sample_step,4]-state[0,4]),np.rad2deg(state[::sample_step,7]-state[0,7])],
              [BLUE,GOLD],(-50.,50.),'关节构型变化 / °',t,['关节 2','关节 5'])
        if k in (0,round(args.fps*min(11.,duration/2))):
            image.save(output.with_name('preview.png' if k else 'first_frame.png'))
        writer.write(cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR))
    writer.release()
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(tmp),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],check=True)
    tmp.unlink()
    import hashlib
    manifest=dict(backend=summary['backend'],source_run=str(args.run.resolve()),
        fps=args.fps,frames=k+1,continuous_simulation_time=True,
        telemetry_sha256=hashlib.sha256((args.run/'telemetry.json').read_bytes()).hexdigest(),
        summary_sha256=hashlib.sha256((args.run/'summary.json').read_bytes()).hexdigest(),
        video_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2))
    print(output)


if __name__=='__main__':main()
