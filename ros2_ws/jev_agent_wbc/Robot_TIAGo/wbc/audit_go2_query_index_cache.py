#!/usr/bin/env python3
"""Bitwise equivalence and alternating-order static CPU query-index benchmark."""
import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial.transform import Rotation


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sources=a.output/'sources';sources.mkdir(parents=True,exist_ok=True)
    if (a.output/'summary.json').exists():raise FileExistsError(a.output/'summary.json')
    original_path=sources/'original_bounded_body_prediction.py'
    assert original_path.is_file(),'freeze original before editing'
    shutil.copy2(a.robot_dir/'bounded_body_prediction.py',sources/'candidate_bounded_body_prediction.py')
    shutil.copy2(a.robot_dir/'wbc_backend.py',sources/'wbc_backend.py')
    shutil.copy2(__file__,sources/Path(__file__).name)
    old=load('original_query_lookup',original_path)
    new=load('candidate_query_lookup',sources/'candidate_bounded_body_prediction.py')
    backend=load('query_cache_backend',sources/'wbc_backend.py')
    rng=np.random.default_rng(8935)
    intervals=[(0.,.2,.4,.6,.8,1.,1.2),(0.,.017,.15,.393,.8013,1.197),
        (0.,.2,np.nextafter(.2,np.inf),.2+5e-11,.2+9e-11,.2+1.2e-10,1.2),
        (0.,0.,.15,.15+5e-11,.51,1.2)]
    contexts=[];comparisons=0;exceptions=0
    for case in range(12):
        body=np.eye(4);body[:3,:3]=Rotation.from_euler('xyz',rng.uniform([-.4,-.3,-3],[.4,.3,3])).as_matrix()
        body[:3,3]=rng.uniform(-1.,1.,3)
        local=np.eye(4);local[:3,3]=[.03,-.02,.07]
        local[:3,:3]=Rotation.from_euler('xyz',[.03,-.02,.01]).as_matrix()
        previous=rng.uniform(-.13,.13,3)
        history=[(5.+(i-9)*.02,previous+np.array([.003,.002,.004])*(i-8))for i in range(9)]
        kwargs=dict(stamp_s=5.,body=body,link=body@local,measured_velocity=rng.uniform(-.16,.16,3),
            previous_applied=previous,input_history=history,prediction_intervals=intervals[case%4],
            pending_mode='measured_hold'if case%2 else'linear_residual')
        before=old.BoundedBodyPrediction(backend,**kwargs);after=new.BoundedBodyPrediction(backend,**kwargs)
        first=previous+rng.uniform(-1,1,3)*np.array([.009,.009,.014])
        future=rng.uniform(-.2,.2,3)
        for name in ('events','delta','delayed_indices','history_times','history_inputs','command_times','root_to_link'):
            np.testing.assert_array_equal(getattr(before,name),getattr(after,name));comparisons+=1
        original_trajectory=before.trajectory(first,future);candidate_trajectory=after.trajectory(first,future)
        for key in original_trajectory:
            np.testing.assert_array_equal(original_trajectory[key],candidate_trajectory[key]);comparisons+=1
        elapsed=list(kwargs['prediction_intervals'])
        for value in kwargs['prediction_intervals']:
            elapsed.extend([np.nextafter(value,np.inf),np.nextafter(value,-np.inf),value+5e-11,value-5e-11])
        elapsed.extend([None,np.nan,np.inf,-np.inf,-.01,1.3,'0.2',Decimal('.2'),Fraction(1,5),
            [0.2],np.asarray(.2),np.asarray([.2,.4]),True,False,np.float32(.2),np.float32(0.),np.longdouble(.2)])
        for value in elapsed:
            returned=[]
            for model in (before,after):
                try:returned.append(('value',model.predict_link(first,future,value)))
                except Exception as exc:returned.append(('error',type(exc).__name__,str(exc)))
            assert returned[0][0]==returned[1][0],repr(value)
            if returned[0][0]=='value':
                np.testing.assert_array_equal(returned[0][1],returned[1][1]);comparisons+=1
            else:assert returned[0]==returned[1],repr(value);exceptions+=1
        for bad_first,bad_future in [(None,future),([np.nan,0,0],future),([0,0],future),([.21,0,0],future),
                (-previous,future),(first,[0,0]),(first,[np.inf,0,0]),(first,[0,.21,0])]:
            errors=[]
            for model in (before,after):
                try:model.predict_link(bad_first,bad_future,0.);errors.append(None)
                except Exception as exc:errors.append((type(exc).__name__,str(exc)))
            assert errors[0]==errors[1] and errors[0]is not None;exceptions+=1
        # Last-nearby-event semantics and returned copies must be preserved.
        for value in kwargs['prediction_intervals']:
            index=int(np.flatnonzero(np.isclose(before.events,value,atol=1e-10,rtol=0))[-1])
            assert after.query_event_indices[float(value)]==index
            result=after.predict_link(first,future,value);result[:]=12345.
            np.testing.assert_array_equal(after.predict_link(first,future,value),original_trajectory['links'][index]);comparisons+=1
        assert before.manifest()==after.manifest()
        contexts.append((kwargs,first,future))

    # The complete scalar query set is exact and reused by the real seven-node
    # provider. Timed calls include no waits; order reverses each ABBA block.
    kwargs,first,future=contexts[0]
    models={'original':old.BoundedBodyPrediction(backend,**kwargs),'candidate':new.BoundedBodyPrediction(backend,**kwargs)}
    queries=tuple(kwargs['prediction_intervals'])
    for model in models.values():
        for query in queries:model.predict_link(first,future,query)
    def elapsed_ms(callback):
        start=time.perf_counter_ns();callback();return(time.perf_counter_ns()-start)/1e6
    def hot(label):
        model=models[label]
        for _ in range(160):
            for query in queries:model.predict_link(first,future,query)
    proposals=[(first+np.array([i*1e-5,-i*1e-5,i*1e-5]),future)for i in range(12)]
    def pipeline(label):
        module=old if label=='original'else new
        model=module.BoundedBodyPrediction(backend,**kwargs)
        for command,target in proposals:
            for query in queries:model.predict_link(command,target,query)
    benchmarks={}
    for name,callback,calls in [('cached_trajectory_exact_query',hot,160*len(queries)),
                              ('construct_plus_12_proposals_7_nodes',pipeline,len(proposals)*len(queries))]:
        blocks=[]
        for block in range(16):
            order=('original','candidate','candidate','original')if block%2==0 else('candidate','original','original','candidate')
            times={'original':[],'candidate':[]}
            for label in order:times[label].append(elapsed_ms(lambda:callback(label)))
            blocks.append(dict(order=list(order),original_ms=float(np.mean(times['original'])),
                               candidate_ms=float(np.mean(times['candidate']))))
        old_ms=float(np.median([b['original_ms']for b in blocks]));new_ms=float(np.median([b['candidate_ms']for b in blocks]))
        benchmarks[name]=dict(calls_per_measurement=calls,blocks=blocks,median_original_ms=old_ms,
            median_candidate_ms=new_ms,median_ratio_candidate_over_original=new_ms/old_ms,
            median_saved_ms=old_ms-new_ms,candidate_faster_blocks=sum(b['candidate_ms']<b['original_ms']for b in blocks))
    report=dict(schema='go2_query_index_cache_independent/v1',equivalent=True,bitwise_comparisons=comparisons,
        equivalent_exception_cases=exceptions,body_history_cases=len(contexts),
        edge_scope=['full roll/pitch/yaw','both pending modes','moving input history','non-grid original nodes',
                    'nearly coincident nodes preserve LAST isclose index','near-ULP exact and fallback',
                    'invalid elapsed/first/future errors','returned independent copies','unchanged trajectory/events/manifest'],
        benchmarks=benchmarks,improvement_demonstrated=all(v['median_candidate_ms']<v['median_original_ms']for v in benchmarks.values()),
        execution='Static CPU only; native elapsed wall execution without deliberate slowdown/waits. No physics, policy dynamics or end-to-end pace claim.',
        frozen_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in sources.iterdir()if p.is_file()})
    (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k!='benchmarks'},indent=2))
    print(json.dumps({k:{key:value for key,value in v.items()if key!='blocks'}for k,v in benchmarks.items()},indent=2))


if __name__=='__main__':main()
