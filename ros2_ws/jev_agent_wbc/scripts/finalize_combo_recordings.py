import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'jev_agent_wbc').is_dir() and (p/'scenarios').is_dir())
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from record_combo_trajectories import ComboReference, digest
from jev_agent_wbc.scenarios import RuntimeScene
from jev_agent_wbc.whole_body_qp import whole_body_kinematics
from jev_agent_wbc.kc_svsdf import CollisionGeometry

out = ROOT/'scenarios/video_combo'
catalog = json.loads((ROOT/'scenarios/combined_catalog.json').read_text())
manifest = dict(videos=[], combined_video='all_scenarios_third_person.mp4', overview_image='third_person_overview.jpg',
    nominal_audit='nominal_audit.json', interference_audit='interference_audit.json',
    robot_display='independent per-action offline nominal WholeBodyQP; animated URDF FK; not PhysX or Agent execution',
    transition_policy='empty windows preserved; no reference curve, robot pose, camera pose or invented arc during gaps; obstacles keep moving',
    playback_rate=1., layout='matching scenarios/video: 1280x720, 30 fps, fixed Isaac observer and fixed detail inset')
panels = []
for case in catalog['scenarios']:
    name = case['name']; video = out/f'{name}_third_person.mp4'
    meta = json.loads((out/f'{name}_third_person_recording.json').read_text())
    assert all(digest(Path(p))==sha for p,sha in meta['source_sha256'].items())
    assert digest(ROOT/'scripts/record_combo_trajectories.py') == meta['script_sha256']
    scene = RuntimeScene(ROOT/'scenarios'/name/'runtime_scene.json')
    ref = ComboReference(ROOT/'scenarios'/name/'task.json',scene.document['subject']['position'])
    gap_count = 0; max_robot_error = 0.; observations = []
    assert len(meta['frames']) == round(case['duration_s']*30)
    for index, frame in enumerate(meta['frames']):
        t=frame['trajectory_time_s']; assert abs(t-index/30)<1e-10
        sample=ref.sample(t)
        if sample is None:
            assert frame['transition_empty'] and frame['phase'] is None
            assert all(frame[k] is None for k in ('robot_state','camera_position','camera_quaternion_xyzw'))
            gap_count+=1
        else:
            assert not frame['transition_empty']
            assert np.allclose(frame['camera_position'],sample[0],atol=1e-9)
            p,_,_=whole_body_kinematics(frame['robot_state'],scene.document['base_z'])
            max_robot_error=max(max_robot_error,float(np.linalg.norm(p-sample[0])))
        truth=scene.obstacles(t)
        assert len(truth)==len(frame['obstacles'])
        for obj,saved in zip(truth,frame['obstacles']):
            assert obj['id']==saved['id'] and np.allclose(obj['center'],saved['center'],atol=1e-9)
            assert np.allclose(obj['velocity'],saved['velocity'],atol=1e-9)
    visible=[f for f in meta['frames'] if not f['transition_empty']]
    states=np.array([f['robot_state'] for f in visible])
    centers=np.array([[o['center'] for o in f['obstacles']] for f in visible])
    radii=np.array([o['radius'] for o in visible[0]['obstacles']])
    clearances=CollisionGeometry().distances_sequence(states,centers,scene.document['base_z'])-radii[None,:,None]
    measured_min=float(clearances.min())
    audit=case['interference_audit']
    assert len(scene.document['obstacles'])==3 and audit['all_obstacles_interfere']
    assert abs(measured_min-audit['nominal_main_min_surface_clearance_m'])<.01
    per_obstacle=clearances.min(axis=(0,2))
    assert np.all(per_obstacle<-.015), (name,per_obstacle)
    contacts=clearances.min(axis=2)<-.003
    assert np.all(contacts.sum(axis=0)>=6)
    assert int(contacts.sum(axis=1).max())==(2 if case['difficulty']=='hard' else 1)
    for frame,distances in zip(visible,clearances.min(axis=2)):
        saved=frame['obstacle_surface_clearances_m']
        assert all(abs(saved[o['id']]-d)<1e-8 for o,d in zip(frame['obstacles'],distances))

    assert gap_count == len(case['transitions'])*59  # Endpoints belong to authored actions.
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration','-of','json',str(video)]))
    stream=probe['streams'][0]
    assert stream['width']==1280 and stream['height']==720 and stream['r_frame_rate']=='30/1'
    assert int(stream['nb_frames'])==len(meta['frames'])
    assert abs(float(probe['format']['duration'])-case['duration_s'])<.001
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
    first_t=round(audit['obstacles'][1]['witness_time_s']*30)/30
    crossing=out/f'{name}_third_person_{first_t:05.2f}.png'
    shutil.copyfile(crossing,out/f'{name}_third_person_crossing.png')
    panels.append(Image.open(crossing).convert('RGB').resize((640,360),Image.LANCZOS))
    # Check actual encoded main-view pixels move during each authored action.
    diffs=[]
    for action in case['main_actions']:
        frames=[]
        for fraction in (.1,.7):
            t=action['start_s']+fraction*action['duration_s']
            raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(t),'-i',str(video),
                '-frames:v','1','-vf','crop=900:480:0:120,scale=300:160','-f','rawvideo','-pix_fmt','rgb24','-'])
            frames.append(np.frombuffer(raw,np.uint8).astype(float))
        difference=float(np.abs(frames[1]-frames[0]).mean());assert difference>.1
        diffs.append(difference)
    manifest['videos'].append(dict(case=name,difficulty=case['difficulty'],obstacle_count=case['obstacle_count'],interference_audit=audit, all_three_obstacles_interfere_verified=True, per_obstacle_recorded_min_surface_clearance_m=per_obstacle.tolist(),file=video.name,**stream,duration=probe['format']['duration'],
        size_bytes=video.stat().st_size,source_trajectory_verified=True,full_decode_pass=True,
        rendered_motion_verified=True,action_pixel_differences=diffs,
        empty_transition_frames=gap_count, max_nominal_camera_error_m=max_robot_error,
        source_sha256=meta['source_sha256']))
    print(f'VERIFIED {name}: {len(meta["frames"])} frames, {gap_count} empty gap frames',flush=True)

playlist=out/'concat_inputs.txt'
playlist.write_text(''.join(f"file '{item['file']}'\n" for item in manifest['videos']))
combined=out/manifest['combined_video']
subprocess.run(['ffmpeg','-v','error','-nostdin','-n','-f','concat','-safe','0','-i',str(playlist),
    '-c','copy','-movflags','+faststart',str(combined)],check=True)
subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(combined),'-f','null','-'],check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
    '-show_entries','stream=nb_frames:format=duration','-of','json',str(combined)]))
manifest.update(combined_duration_s=float(probe['format']['duration']),combined_frames=int(probe['streams'][0]['nb_frames']),combined_decode_pass=True)
assert manifest['combined_duration_s']==sum(c['duration_s'] for c in catalog['scenarios'])
assert manifest['combined_frames']==sum(int(v['nb_frames']) for v in manifest['videos'])
overview=Image.new('RGB',(1280,360*((len(panels)+1)//2)))
for i,panel in enumerate(panels):overview.paste(panel,((i%2)*640,(i//2)*360))
overview.save(out/manifest['overview_image'],quality=94)
(out/'third_person_videos.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
source=out/'source_snapshot';source.mkdir(exist_ok=True)
for name in ('record_combo_trajectories.py','record_scenario_trajectories.py','author_graded_combo_scenarios.py','design_combo_interferences.py','finalize_combo_recordings.py'):
    shutil.copyfile(ROOT/'scripts'/name,source/name)
shutil.copyfile(ROOT/'scenarios/combined_catalog.json',source/'combined_catalog.json')
for name in catalog['cases']:
    folder=source/name;folder.mkdir(exist_ok=True)
    for file in ('task.json','runtime_scene.json'):
        shutil.copyfile(ROOT/'scenarios'/name/file,folder/file)
shutil.copyfile(__file__,out/'finalize_recordings.py')
(out/'README.txt').write_text('''六个分级组合运镜场景设计预览

六个独立视频和一个 134 秒合辑；1280×720、30 fps、H.264，按 1× 场景时间播放。
与 scenarios/video 相同的 Isaac Sim 第三人称主视角、固定局部视角、路径和时间标注。

六个场景均为低、中、高各一个障碍物，共三个。中心高度：低 0.16 m，中 0.48 m，
高约 0.76–0.86 m（按各镜头的上部机械臂/相机高度设置）。三者分别横穿底盘、
中部机械臂和上部机械臂/相机，不能只在空间上经过而错开时间。
01/02 简单：错开的浅干涉，目标穿透深度约 0.025 m。
03/04 中等：错开的较深干涉，目标穿透深度约 0.055 m。
05/06 困难：目标穿透深度约 0.085 m，包含两个障碍物同时干涉。
所有 18 个障碍物都通过同时刻整机几何干涉检查，以及成片逐帧复核。
interference_audit.json 提供 10 ms 检查的见证时间、部件、最小 signed clearance、
干涉时段及持续时间。净距为负代表球形障碍物表面与机器人碰撞几何发生重叠。
这里是无避障回放，所以会看到穿插；这正是为后续 Agent 避障安排的干涉事件。
衔接留空，不将该窗口算入干涉证据；障碍物与障碍物/主体之间仍保留正间距。

这是原始场景设计回放，不是 Agent 闭环避障或 PhysX 执行结果。
青色相机标记精确跟随各主动作的参考轨迹。机器人使用每个主动作独立的离线无避障
WholeBodyQP 运动示意；新动作的起始机器人位姿用静态 IK 单独放置，不生成衔接运动。
nominal_audit.json 记录逐段位置与姿态误差，不代表执行或安全验收通过。

各 2 秒衔接窗口无轨迹、无占位 orbit。窗口内部隐藏参考相机和机器人，显示
“衔接留空 · 待 Agent 设计 arc”；障碍物按绝对场景时间连续运动。下一动作开始时
重新显示相机及该动作的机器人示意。整段视频没有暂停、慢放或预先设计的连接路径。
主动作路径线也在衔接处断开。

*_recording.json 保存逐帧位姿、障碍物和源文件哈希，空白帧的相机/机器人位姿为 null。
third_person_videos.json 保存全片解码、尺寸、时长、逐帧源轨迹与空白窗口检查结果。
source_snapshot 保存本次场景输入和录制脚本；原 scenarios/video 未更改。video_combo 在本次生成前已清空，上一版六个场景及视频已移出并归档，位置见 previous_version.json。
''')
print(f'COMPLETE: {combined}, {manifest["combined_frames"]} frames',flush=True)
