"""Native input I must not survive parking/recovery/collision overrides."""
import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


runner=module('z_wiring_runner',ROOT/'Robot_Go2/adaptive_runner.py')
feedback=module('z_wiring_helper',ROOT/'Robot_Go2/world_z_feedback.py')
backend=module('z_wiring_backend',ROOT/'Robot_Go2/wbc_backend.py')
coordinator=module('z_wiring_coordinator',ROOT/'Robot_Go2/camera_coordinator.py')


def sample(controller,step):
    reference=np.eye(4);reference[:3,3]=[.5,0,.5]
    measured=reference.copy();measured[2,3]+=.02
    return controller.feedback_native_command(backend,reference,measured,np.eye(4),np.eye(4),
        task_id='accepted',stamp_s=step*.02,now_s=step*.02,dt_s=.02,sample_seq=step,
        domain=backend.PolicyDomain())


@pytest.mark.parametrize('reason',['native_parking_hold','native_domain_recovery','collision_freshness_veto',
    'original_camera_support_rejected','actual_native_domain_failure'])
def test_outer_override_reverts_new_integral_and_preserves_actual_domain(reason):
    controller=feedback.WorldZFeedback(enabled=True)
    native,checked=sample(controller,0)
    assert controller.integral_m<0
    actual=dict(status='inside_outer_domain_unverified',actual=True)
    result=runner.finalize_native_feedback(controller,checked,actual,[reason])
    assert result is actual and result['actual']
    assert controller.integral_m==0
    report=result['world_z_feedback']
    assert report['externally_rolled_back'] and not report['integral_updated']
    assert report['external_rollback_reason']==reason
    _,duplicate=sample(controller,0)
    assert not duplicate['world_z_feedback']['measurement_fresh_and_clock_valid']
    assert controller.integral_m==0


def test_real_native_parking_hold_rolls_back_integrator_transaction():
    controller=feedback.WorldZFeedback(enabled=True);parking=coordinator.NativeCameraParking()
    native,checked=sample(controller,0)
    reference=np.eye(4);reference[:3,3]=[.5,0,.5]
    held,report=parking.update(native,reference,reference,np.eye(4),np.zeros(3),parking=True)
    assert report['active']
    actual=backend.PolicyDomain().check(held)
    result=runner.finalize_native_feedback(controller,checked,actual,['native_parking_hold'])
    assert result['status']=='inside_outer_domain_unverified'
    assert controller.integral_m==0


def test_real_native_domain_recovery_rolls_back_integral():
    controller=feedback.WorldZFeedback(enabled=True);domain=backend.PolicyDomain()
    recovery=coordinator.NativeDomainRecovery(domain,backend)
    recovery.seed(np.array([.5,0,.5,1,0,0,0]))
    native,checked=sample(controller,0)
    outside=native.copy();outside[0]=-1
    held,brake,report=recovery.update(outside,np.eye(4),np.eye(4),0,np.zeros(3),.02)
    assert report['active'] and brake is not None
    actual=domain.check(held)
    result=runner.finalize_native_feedback(controller,checked,actual,['native_domain_recovery'])
    assert result['status']=='inside_outer_domain_unverified'
    assert result['world_z_feedback']['externally_rolled_back']
    assert controller.integral_m==0


def test_no_override_commits_integral_and_legacy_path_has_no_report():
    controller=feedback.WorldZFeedback(enabled=True)
    native,checked=sample(controller,0);before=controller.integral_m
    actual=backend.PolicyDomain().check(native)
    result=runner.finalize_native_feedback(controller,checked,actual,[])
    assert controller.integral_m==before<0
    assert result['world_z_feedback']['native_correction_accepted']
    legacy=dict(status='inside_outer_domain_unverified')
    assert runner.finalize_native_feedback(None,{},legacy,['any'])==legacy
    assert 'world_z_feedback' not in legacy


def test_wire_uses_measured_absolute_stamp_and_global_sample_seq():
    tree=ast.parse((ROOT/'Robot_Go2/adaptive_runner.py').read_text())
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='feedback_native_command' and isinstance(n.func.value,ast.Name) and n.func.value.id=='world_z_feedback')
    args={n.arg:ast.unparse(n.value) for n in call.keywords}
    assert args['stamp_s']=='observation.stamp_s'
    assert args['now_s']=='stamp' and args['sample_seq']=='step' and args['dt_s']=='dt'
    assert args['task_id']=='session.task_id'
    assert "collision_evidence['status'] == 'pass'" in args['integration_allowed']
    assert "requested_domain['status'] == 'inside_outer_domain_unverified'" in args['integration_allowed']


def test_native_finalizer_runs_after_every_outer_guard_before_policy_tensor():
    source=(ROOT/'Robot_Go2/adaptive_runner.py').read_text()
    begin=source.index('feedback_overrides=[]')
    end=source.index('ee=torch.tensor(native',begin)
    block=source[begin:end]
    assert block.index("feedback_overrides.append('native_parking_hold')")<block.index('corrected=finalize_native_feedback(')
    assert block.index("feedback_overrides.append('native_domain_recovery')")<block.index('corrected=finalize_native_feedback(')
    assert block.index("feedback_overrides.append('collision_freshness_veto')")<block.index('corrected=finalize_native_feedback(')
    assert block.index("feedback_overrides.append('original_camera_support_rejected')")<block.index('corrected=finalize_native_feedback(')
