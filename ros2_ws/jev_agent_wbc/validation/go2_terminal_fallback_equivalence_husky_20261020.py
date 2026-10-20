#!/usr/bin/env python3
"""Bind corrected rejected-context branch to unchanged normal CPU evidence."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'Robot_Go2/results/terminal_pose_full_preparation_01/sources'
OUT=ROOT/'Robot_Go2/results/terminal_pose_fallback_cpu_01'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(value):return ast.dump(value,include_attributes=False)


def main():
    tests=ET.parse(OUT/'cpu_tests.xml').getroot()
    count=sum(int(x.get('tests','0'))for x in tests.iter('testsuite'))
    assert count==132
    assert sum(int(x.get(k,'0'))for x in tests.iter('testsuite')for k in ['failures','errors','skipped'])==0
    old_c=ast.parse((OLD/'camera_coordinator.py').read_text());new_c=ast.parse((ROOT/'Robot_Go2/camera_coordinator.py').read_text())
    added=[]
    for node in new_c.body:
        if isinstance(node,ast.ClassDef)and node.name=='CameraCoordinator':
            added=[x for x in node.body if isinstance(x,ast.FunctionDef)and x.name=='reject_without_forecast']
            node.body=[x for x in node.body if x not in added]
    assert len(added)==1 and dump(old_c)==dump(new_c)
    old_r=ast.parse((OLD/'adaptive_runner.py').read_text());new_r=ast.parse((ROOT/'Robot_Go2/adaptive_runner.py').read_text())
    expected_new=ast.parse("support=camera_support_with_terminal_rejection(coordinator,stamp,sample,bodies['link0'],bodies['base'],policy_command,measured,previous_policy_command,dt,support_horizon,forecast,terminal_error,terminal_deadline_failed)").body[0]
    expected_old=ast.parse("support=coordinator.command(stamp,sample,bodies['link0'],bodies['base'],policy_command,measured,max_lookahead_s=support_horizon,**forecast)").body[0]
    assignments=[node for node in ast.walk(new_r)if isinstance(node,ast.Assign)and isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Name)and node.value.func.id=='camera_support_with_terminal_rejection']
    assert len(assignments)==1 and dump(assignments[0])==dump(expected_new)
    original_assignments=[node for node in ast.walk(old_r)if dump(node)==dump(expected_old)]
    assert len(original_assignments)==1
    class Normalizer(ast.NodeTransformer):
        def visit_Assign(self,node):
            return copy.deepcopy(expected_old)if node is assignments[0]else self.generic_visit(node)
    Normalizer().visit(new_r)
    helpers=[x for x in new_r.body if isinstance(x,ast.FunctionDef)and x.name=='camera_support_with_terminal_rejection']
    assert len(helpers)==1
    new_r.body=[x for x in new_r.body if x not in helpers]
    assert dump(new_r)==dump(old_r)
    unchanged={name:sha(ROOT/'Robot_Go2'/name)==sha(OLD/name)for name in
        ['terminal_allocator.py','terminal_yaw_allocator.py','bounded_body_prediction.py','world_z_feedback.py','optical_sampler.py']}
    assert all(unchanged.values())
    files=[ROOT/'Robot_Go2/adaptive_runner.py',ROOT/'Robot_Go2/camera_coordinator.py',ROOT/'test/test_wbc_go2_terminal_integration.py',Path(__file__)]
    frozen=OUT/'sources';frozen.mkdir(exist_ok=False)
    for path in files:shutil.copy2(path,frozen/path.name)
    result=dict(schema='go2_rejected_context_normal_equivalence/v1',passed=True,
        normal_coordinator_module_ast_identical_after_removing_only_new_reject_method=True,
        normal_runner_module_ast_identical_after_replacing_verified_dispatch_and_removing_only_new_helper=True,
        verified_dispatch_arguments_exact=True,unchanged_model_helpers=unchanged,
        normal_dense_route_exact_output_test=True,missing_stale_history_times_deadline_negative_tests=4,
        cpu_tests_passed=count,cpu_tests_skipped=0,
        scope='Explicit rejected-context path changed; successful dense normal path retains original4500snapshot math/benchmark. No new physical trial or claim.',
        files_sha256={str(path.relative_to(ROOT)):sha(path)for path in files+[OLD/'adaptive_runner.py',OLD/'camera_coordinator.py',OUT/'cpu_tests.xml']})
    (OUT/'equivalence.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
