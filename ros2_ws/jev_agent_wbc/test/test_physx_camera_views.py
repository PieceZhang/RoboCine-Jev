"""PhysX camera routing and capture clocks; no Isaac runtime is required."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from render_physx_demos import film_views
from run_isaac import RGBPair, arguments


def test_stabilized_film_uses_live_reference_and_never_displays_raw_camera():
    reference=dict(protocol='original_reference_floating_camera_same_scene_time/v1',
                   valid=True,same_render_pass=True)
    views=film_views('stabilized',reference)
    assert [v['stream'] for v in views]==['overview','reference','stabilized']
    assert views[2]['title']=='增稳相机（Agent输入）'
    assert [v['agent_input'] for v in views]==[False,False,True]


def test_untrusted_reference_cannot_be_presented_as_new_agent_comparison():
    with pytest.raises(ValueError,match='verified live'):
        film_views('stabilized',dict(protocol='static_reference_separate_pass/v2',valid=True))


def test_physx_default_camera_is_stabilized(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['run_isaac.py','--output',str(tmp_path)])
    assert arguments().camera_output=='stabilized'


def test_reference_raw_and_stabilized_share_one_capture_clock(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'pxr',SimpleNamespace(Gf=SimpleNamespace(Vec2f=lambda *v:v)))
    camera=object.__new__(RGBPair)
    camera.camera_output=SimpleNamespace(source='stabilized')
    camera.sim=SimpleNamespace(current_time=1.2)
    camera.root=tmp_path;camera.frame_metadata=[dict(frame=0)]
    camera.cameras={key:None for key in ('execution','reference','overview','stabilized')}
    clipping=SimpleNamespace(Set=lambda value:None)
    camera.camera_prims={'stabilized':SimpleNamespace(GetClippingRangeAttr=lambda:clipping)}
    camera.camera_publisher=None;camera.reference_frames_written=0
    poses={};captures=[]
    camera.pose=lambda key,p,r:poses.update({key:(np.array(p),np.array(r))})
    def capture(key,index,advance_render=True):
        captures.append((key,advance_render))
        return np.zeros((4,4,3),np.uint8)
    camera.capture_rgb=capture
    camera.visibility=lambda p,r:dict(available=True)
    rp=np.array([1.,0.,.8]);ap=np.array([.7,.1,.8]);sp=np.array([.75,.08,.8]);rotation=np.eye(3)
    camera.capture(.2,rp,rotation,ap,rotation,{},stabilized_pose=(sp,rotation))
    assert captures==[('execution',True),('overview',False),('reference',False),('stabilized',False)]
    np.testing.assert_array_equal(poses['reference'][0],rp)
    row=camera.frame_metadata[-1]
    assert row['reference_scene_time_s']==row['t']==.2
    assert row['physical_time_before_capture']==row['physical_time_after_capture']==1.2
    np.testing.assert_array_equal(row['camera_position'],sp)
    np.testing.assert_array_equal(row['actual_camera_position'],ap)
    assert camera.reference_frames_written==1


def test_close_never_rerenders_reference_at_the_final_scene_time(tmp_path,monkeypatch):
    from jev_agent_wbc import camera_output
    monkeypatch.setattr(camera_output,'write_three_camera_video',lambda *args:dict(frames=1))
    camera=object.__new__(RGBPair);camera.root=tmp_path
    camera.frame_metadata=[dict(frame=0,t=0.)];camera.reference_frames_written=1
    camera.recording=True;camera.args=SimpleNamespace(skip_reference_video=False,record_fps=12.)
    camera.writers={};camera.annotators={};camera.products={};camera.camera_publisher=None
    camera.capture_rgb=lambda *args:pytest.fail('reference must have been captured during execution')
    camera.close()
    audit=json.loads((tmp_path/'reference_rendering.json').read_text())
    assert audit['valid'] and audit['same_render_pass'] and not audit['actor_visibility_modified']
    assert audit['camera_trajectory']=='original_reference' and audit['physics_body'] is False
