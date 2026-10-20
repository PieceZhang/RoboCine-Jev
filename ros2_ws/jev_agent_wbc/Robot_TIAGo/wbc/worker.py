#!/usr/bin/env python3
"""Isolated genuine Giskard worker; bounded one-packet physical request channel."""
import argparse
import json
import os
from pathlib import Path
import pickle
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'wbc')]
os.environ['XDG_CACHE_HOME'] = str(ROOT/'wbc/cache')
os.environ['MPLCONFIGDIR'] = str(ROOT/'wbc/cache/matplotlib')
os.environ['JAX_PLATFORMS']='cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd',type=int,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--robot-dir',type=Path,default=ROOT)
    parser.add_argument('--spec',type=Path)
    parser.add_argument('--backend',choices=['giskard-executor','giskard-core'],default='giskard-core')
    parser.add_argument('--mesh-cache',type=Path)
    parser.add_argument('--sensor-config',type=Path)
    args = parser.parse_args()
    channel = socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET,fileno=args.fd)
    sys.path[:0]=[str(Path(__file__).resolve().parent),str(args.robot_dir)]
    import wbc_backend
    wbc_backend.ROOT=args.robot_dir
    from native_solver import NativeGiskardSolver
    spec = json.loads((args.spec or args.robot_dir/'robot.json').read_text())
    metadata = json.loads(args.metadata.read_text())
    if args.backend=='giskard-executor':
        from full_framework import FullGiskardExecutor
        solver=FullGiskardExecutor(spec,metadata,args.mesh_cache or args.robot_dir/'wbc/cache/world_binary_mesh',sensor_config=args.sensor_config)
    else:
        solver=NativeGiskardSolver(spec,metadata)
    channel.send(pickle.dumps(dict(kind='ready',pid=os.getpid(),solver_commit=solver.commit,
        backend=args.backend,model_manifest=getattr(solver,'model_manifest',None),python=sys.version)))
    while True:
        data = channel.recv(512*1024)
        if not data:
            break
        request = pickle.loads(data)
        if request.get('kind')=='shutdown':
            break
        started = time.perf_counter()
        try:
            result = solver.solve(request)
            result.update(kind='command',ok=True)
        except Exception as exc:
            result = dict(kind='command',ok=False,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        result.update(sequence=request['sequence'],sample_s=request['sample_s'],phase=request['phase'],
            valid_until_s=request['sample_s']+.04,worker_wall_s=time.perf_counter()-started)
        # NumPy2 pickles reference numpy._core, absent in Isaac's NumPy1.26.
        # Return ordinary Python values across the isolated ABI boundary.
        for key in ('command','upstream_proposal'):
            if key in result and hasattr(result[key],'tolist'):result[key]=result[key].tolist()
        channel.send(pickle.dumps(result,protocol=5))


if __name__=='__main__':
    main()
