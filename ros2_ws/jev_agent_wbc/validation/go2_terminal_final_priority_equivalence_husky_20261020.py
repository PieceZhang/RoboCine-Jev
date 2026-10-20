#!/usr/bin/env python3
"""Verify only final rejected-context priority changed after prepared candidate02."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'Robot_Go2/results/terminal_pose_full_preparation_02/sources'
OUT=ROOT/'Robot_Go2/results/terminal_pose_final_priority_cpu_01'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(node):return ast.dump(node,include_attributes=False)


def main():
    tests=ET.parse(OUT/'cpu_tests.xml').getroot();count=sum(int(x.get('tests','0'))for x in tests.iter('testsuite'))
    assert count==136 and sum(int(x.get(k,'0'))for x in tests.iter('testsuite')for k in ['failures','errors','skipped'])==0
    old=ast.parse((OLD/'adaptive_runner.py').read_text());new=ast.parse((ROOT/'Robot_Go2/adaptive_runner.py').read_text())
    def branch(tree,name):
        candidates=[node for node in ast.walk(tree)if isinstance(node,ast.If)and any(isinstance(x,ast.Call)and
            ((isinstance(x.func,ast.Name)and x.func.id==name)or(isinstance(x.func,ast.Attribute)and x.func.attr==name))for x in ast.walk(node))]
        return min(candidates,key=lambda node:len(list(ast.walk(node))))
    before=branch(old,'guard_applied');after=branch(new,'final_terminal_body_command')
    expected=ast.parse('issued,forced_brake=final_terminal_body_command(policy_command,previous_policy_command,dt,terminal_context,terminal_error,terminal_deadline_failed)').body[0]
    assert dump(after.body[0])==dump(expected)
    normalized=copy.deepcopy(after);normalized.test=copy.deepcopy(before.test)
    normalized.body[0]=copy.deepcopy(before.body[0])
    removed=0;metadata=0;kept=[]
    for statement in normalized.body:
        if isinstance(statement,ast.If)and isinstance(statement.test,ast.Name)and statement.test.id=='forced_brake':removed+=1;continue
        if isinstance(statement,ast.Expr)and isinstance(statement.value,ast.Call):
            call=statement.value
            for keyword in call.keywords[:]:
                if keyword.arg=='rejected_forecast_final_brake':call.keywords.remove(keyword);metadata+=1
        kept.append(statement)
    normalized.body=kept
    assert removed==1 and metadata==1 and dump(normalized)==dump(before)
    class Replace(ast.NodeTransformer):
        def visit_If(self,node):return copy.deepcopy(before)if node is after else self.generic_visit(node)
    Replace().visit(new)
    functions=[node for node in new.body if isinstance(node,ast.FunctionDef)and node.name=='final_terminal_body_command']
    assert len(functions)==1
    new.body=[node for node in new.body if node not in functions]
    assert dump(old)==dump(new)
    unchanged={name:sha(ROOT/'Robot_Go2'/name)==sha(OLD/name)for name in
        ['camera_coordinator.py','terminal_allocator.py','terminal_yaw_allocator.py','bounded_body_prediction.py','world_z_feedback.py','optical_sampler.py']}
    assert all(unchanged.values())
    files=[ROOT/'Robot_Go2/adaptive_runner.py',ROOT/'test/test_wbc_go2_terminal_integration.py',Path(__file__)]
    frozen=OUT/'sources';frozen.mkdir(exist_ok=False)
    for path in files:shutil.copy2(path,frozen/path.name)
    report=dict(schema='go2_final_rejected_context_priority/v1',passed=True,cpu_tests_passed=count,cpu_tests_skipped=0,
        normal_runner_module_ast_equivalent_except_verified_final_dispatch_and_false_diagnostic_field=True,
        unchanged_coordinator_and_model_helpers=unchanged,
        actual_final_route_negative_tests=3,actual_fresh_normal_projection_exact_test=True,
        scope='Only rejected terminal-error or latched-deadline body priority changed. Final original slew brake overrides later body recovery while keeping its legal native7; normal dense path unchanged. Not a physical trial.',
        files_sha256={str(path.relative_to(ROOT)):sha(path)for path in files+[OLD/'adaptive_runner.py',OUT/'cpu_tests.xml']})
    (OUT/'equivalence.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
