#!/usr/bin/env python3
"""Offline, truth-assisted authoring witnesses; never an Agent evaluation.

Try phase-preserving smooth clocks on an obstacle-free body path. Every found
witness is integrated with physical CPU commands and checked for actuator,
whole-body, self-collision and unchanged shot acceptance. Failure to find a
witness does NOT prove that an episode is impossible.
"""
import argparse
from contextlib import redirect_stdout
import io
import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from CineGym.episode import Episode,write_json
from CineGym.authoring import WithTerminalHold,audit_nominal
from CineGym.obstacles import motion_bounds
from jev_agent_wbc.nominal import nominal_rollout
from jev_agent_wbc.whole_body_qp import Limits
from jev_agent_wbc.kc_svsdf import advance,CollisionGeometry
from jev_agent_wbc.actuation import advance_command
from jev_agent_wbc.self_collision import SelfCollision
from jev_agent_wbc.jvs_kinematics import JOINT_LOWER,JOINT_UPPER


def witness(episode,output):
    from CineGym.solvability import timing_pool,finish_witness
    reference=episode.reference;scene=episode.scene;runtime=episode.runtime
    with redirect_stdout(io.StringIO()):
        ts,xs,failure=nominal_rollout(WithTerminalHold(reference),scene['initial_state'],scene['base_z'])
    if failure:return dict(status='unverified',reason=failure)
    recorded=episode.metadata.get('avoidance_solvability',{})
    clocks=recorded.get('clock') if recorded.get('status')=='certified_cpu_timing_witness' else None
    # Older prototype metadata stored one common clock instead of per-phase clocks.
    if clocks and isinstance(clocks[0],(int,float)):
        clocks=[clocks]*len(reference.task['shot_phase'])
    pool=timing_pool(reference,scene,ts,xs,selected_clocks=clocks)
    curves=list(runtime.curves.values());geometry=CollisionGeometry();own=SelfCollision()
    for i,item in enumerate(pool):
        proof=finish_witness(reference,scene,pool,[i],curves)
        if proof is None or proof['continuous_external_lower_m']<.025:continue
        # Independently integrate the saved commands using the actual runner's
        # constant-command advance(), not the vectorized authoring integrator.
        states=[np.asarray(scene['initial_state'])]
        for command in item['controls']:states.append(advance(states[-1],command,.02))
        states=np.asarray(states)
        if not np.allclose(states,item['states'],atol=1e-10,rtol=0.):
            return dict(status='unverified',reason='independent CPU replay mismatch')
        np.savez_compressed(output,times=item['times'],states=states,commands=item['controls'],progress=item['progress'])
        import hashlib
        return dict(proof,status='found',witness_file=str(output),
            file_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            independently_replayed_cpu_commands=True)
    return dict(status='not_found',actuator_feasible_attempts=len(pool),
        scope='Restricted timing family found no certified witness; global solvability remains unknown')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('suite',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);results=[]
    for name in json.loads((a.suite/'catalog.json').read_text())['cases']:
        e=Episode.load(a.suite/name);result=dict(case=name,seed=e.metadata['seed'],difficulty=e.metadata['difficulty'],episode_sha256=e.metadata['sha256'],**witness(e,a.output/(name+'.npz')))
        results.append(result);write_json(a.output/'report.json',results);print(name,result['status'],flush=True)

if __name__=='__main__':main()
