#!/usr/bin/env python3
"""Compare the current Ridgeback branch with its actual FULL01 frozen sources.

This is a CPU source/input-output equivalence audit, not another physical run.
It neither imports the new runtime (which starts Kit) nor modifies old results.
"""
import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT.parent

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(tree):return ast.dump(tree,include_attributes=False)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class OmniBranch(ast.NodeTransformer):
    """Partial evaluation of the fixed, verified `omni=True` model selector."""
    def visit_Name(self,node):
        return ast.Constant(True) if node.id=='omni' and isinstance(node.ctx,ast.Load) else node
    def visit_UnaryOp(self,node):
        node=self.generic_visit(node)
        if isinstance(node.op,ast.Not) and isinstance(node.operand,ast.Constant):return ast.Constant(not node.operand.value)
        return node
    def visit_BoolOp(self,node):
        node=self.generic_visit(node)
        if isinstance(node.op,ast.Or) and any(isinstance(v,ast.Constant) and v.value is True for v in node.values):return ast.Constant(True)
        if isinstance(node.op,ast.And) and any(isinstance(v,ast.Constant) and v.value is False for v in node.values):return ast.Constant(False)
        node.values=[v for v in node.values if not isinstance(v,ast.Constant) or v.value is not (False if isinstance(node.op,ast.Or) else True)]
        return node.values[0] if len(node.values)==1 else node
    def visit_If(self,node):
        node=self.generic_visit(node)
        return (node.body if node.test.value else node.orelse) if isinstance(node.test,ast.Constant) else node
    def visit_IfExp(self,node):
        node=self.generic_visit(node)
        return (node.body if node.test.value else node.orelse) if isinstance(node.test,ast.Constant) else node

class AuditOnly(ast.NodeTransformer):
    """Remove only acceptance scalars; never drop actions, steps or guards."""
    def visit_Assign(self,node):
        # Tracking is only read by phase JSON passed=tracking, never control flow.
        if any(isinstance(t,ast.Name) and t.id=='tracking' for t in node.targets):return None
        return self.generic_visit(node)
    def visit_If(self,node):
        if (node.body and isinstance(node.body[0],ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='metrics' for t in node.body[0].targets)
                and isinstance(node.body[0].value,ast.Call)
                and isinstance(node.body[0].value.func,ast.Attribute)
                and node.body[0].value.func.attr=='camera_task_metrics'):
            assert len(node.body)==2 and not node.orelse
            return None
        return self.generic_visit(node)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'results/ocs2_acceptance_01')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    provenance=json.loads((args.run/'provenance.json').read_text())
    sources=provenance['sources'];pairs={}
    for name in ['ocs2_wheeled_runtime.py','ocs2_wheeled_safety.py']:
        key=next(k for k in sources if Path(k).name==name)
        old=args.run/'sources'/sources[key]['snapshot'];new=PACKAGE/'scripts'/name
        assert digest(old)==sources[key]['sha256']
        pairs[name]=(old,new)
        (args.output/(name+'.diff')).write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile=str(old),tofile=str(new))))
    old_run,new_run=[ast.parse(p.read_text()) for p in pairs['ocs2_wheeled_runtime.py']]
    def specialized(module):
        run=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        return dump(AuditOnly().visit(OmniBranch().visit(run)))
    specialized_equal=specialized(old_run)==specialized(new_run)
    old_safety,new_safety=[ast.parse(p.read_text()) for p in pairs['ocs2_wheeled_safety.py']]
    inactive={'SkidSteerVelocityServo','terminal_reference_window','camera_task_metrics'}
    def active(module):
        return dump(ast.Module(body=[n for n in module.body if not isinstance(n,(ast.ClassDef,ast.FunctionDef)) or n.name not in inactive],type_ignores=[]))
    specialized_run=AuditOnly().visit(OmniBranch().visit(next(n for n in ast.parse(pairs['ocs2_wheeled_runtime.py'][1].read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')))
    unused_helpers=not any(isinstance(n,ast.Attribute) and n.attr in inactive for n in ast.walk(specialized_run))
    safety_equal=active(old_safety)==active(new_safety)
    # ALL runtime code outside run(), including source freezing and supervisor,
    # must be unchanged; run() is proved equal after the fixed model selector.
    def outside_run(module):
        return dump(ast.Module(body=[n for n in module.body if not isinstance(n,ast.FunctionDef) or n.name!='run'],type_ignores=[]))
    outside_equal=outside_run(old_run)==outside_run(new_run)
    oldmod=load(pairs['ocs2_wheeled_safety.py'][0],'full01_safety')
    newmod=load(pairs['ocs2_wheeled_safety.py'][1],'current_safety')
    model=json.loads((ROOT/'ocs2/model.json').read_text());spec=json.loads((ROOT/'ocs2/backend.json').read_text())
    assert spec['base_model']=='omnidirectional'
    limits=np.array([[j['lower'],j['upper']] for j in model['joints']])
    before=oldmod.HardCommandGuard(True,limits[:,0],limits[:,1]);after=newmod.HardCommandGuard(True,limits[:,0],limits[:,1])
    max_delta=0.;audits_equal=True;count=0
    with (args.run/'samples.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);indices=[row['joint_names'].index(n) for n in spec['arm_joints']]
            q=np.asarray(row['q'])[indices];qd=np.asarray(row['qd'])[indices];brake=row['phase']=='stop'
            a=before.command(row['command'],q,qd,brake=brake);b=after.command(row['command'],q,qd,brake=brake)
            max_delta=max(max_delta,float(np.max(np.abs(a-b))));audits_equal &= before.last_audit==after.last_audit;count+=1
    unchanged={}
    for original,entry in sources.items():
        if Path(original).name in pairs:continue
        path=Path(original)
        unchanged[original]=bool(path.is_file() and digest(path)==entry['sha256'])
    native_binary=(ROOT/'ocs2/.cache/backend/camera_mpc_worker')
    worker_equal=digest(native_binary)==provenance['worker_sha256']
    guard_equal=digest(ROOT/'ocs2/.cache/backend/libmeasured_collision_guard.so')==digest(args.run/'sources/libmeasured_collision_guard.so')
    camera=[json.loads(line) for line in (args.run/'samples.jsonl').open() if '"case": "ocs2_camera"' in line]
    all_rows=[json.loads(line) for line in (args.run/'samples.jsonl').open()]
    task=json.loads((args.run/'task.json').read_text())
    metrics=newmod.camera_task_metrics([r['position_error_m'] for r in camera],[r['orientation_error_rad'] for r in camera],all_rows[-1]['position_error_m'],all_rows[-1]['orientation_error_rad'],task['position_tolerance_m'],task['orientation_tolerance_rad'])
    original_default_camera_gate=bool(metrics['position_max_m']<task['position_tolerance_m'] and metrics['orientation_max_rad']<task['orientation_tolerance_rad'])
    hashes={name:dict(physical_frozen_sha256=digest(old),current_sha256=digest(new)) for name,(old,new) in pairs.items()}
    result=dict(schema='ocs2_ridgeback_branch_equivalence/v1',scope='CPU source/IO equivalence only; FULL01 remains the actual physical evidence, not a new physical run',robot='ridgeback',passed=bool(specialized_equal and safety_equal and outside_equal and unused_helpers and count and max_delta==0 and audits_equal and all(unchanged.values()) and worker_equal and guard_equal and original_default_camera_gate and metrics['final_task_tolerance_pass']),
                new_summary_metrics_recomputed_from_original_raw=metrics,new_default_tracking_gate=original_default_camera_gate,control_equivalence_scope='Exact omni control AST after removing acceptance-only tracking assignments and the appended post-run camera_task_metrics phase; new summary is independently recomputed above and passes on FULL01 raw data',fixed_model_selector=spec['base_model'],runtime_omni_branch_ast_equal=specialized_equal,runtime_outside_run_ast_equal=outside_equal,safety_except_unexecuted_or_audit_only_definitions_ast_equal=safety_equal,
                inactive_definitions=sorted(inactive),inactive_definitions_unreferenced_by_ridgeback=unused_helpers,measured_rows_replayed=count,maximum_guard_command_delta=max_delta,guard_audit_values_equal=audits_equal,other_executed_sources_unchanged=unchanged,native_worker_binary_equal=worker_equal,native_guard_binary_equal=guard_equal,source_pairs=hashes,
                branch_reason='Ridgeback sets omni=True; the new endpoint-hold controller and terminal_reference_window are non-omni only. camera_task_metrics is post-run acceptance only; all body commands, solver requests, actuator mapping and strict holds stay identical. The exact original tracking loop remains in the omni branch; skid_servo=None; skid_mapping=None. The only runtime deltas occur in the else-only skid_servo.command call and its conditional non-omni metadata. SkidSteerVelocityServo is not constructed or executed by Ridgeback.',
                script_sha256=digest(__file__),physical_provenance_sha256=digest(args.run/'provenance.json'))
    (args.output/'equivalence.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0 if result['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
