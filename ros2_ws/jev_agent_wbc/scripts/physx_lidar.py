"""PhysX sensor transport. All rays retain closest-hit occlusion and source time.

The native adapter is deliberately restricted to the tested Kit 107.3 ABI.
Unsupported installations must select --lidar-backend scalar. Native startup is checked
against the public Python API before the live clock is armed.
"""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def load_native(output):
    import omni.kit.app
    manager = omni.kit.app.get_app().get_extension_manager()
    extension = manager.get_enabled_extension_id('omni.physx')
    if not extension or not extension.startswith('omni.physx-107.3.26'):
        raise RuntimeError(f'native LiDAR requires verified omni.physx 107.3.26, got {extension}')
    import isaacsim
    kit = Path(isaacsim.__file__).resolve().parent/'kit'
    source = Path(__file__).parent/'native/physx_ray_batch.cpp'
    includes = [Path(sysconfig.get_paths()['include']),kit/'dev/include',
                Path(sys.prefix)/'lib/python3.11/site-packages/torch/include']
    digest = hashlib.sha256(source.read_bytes()+sys.version.encode()+extension.encode()).hexdigest()[:16]
    folder = Path(output).resolve().parent/'native_cache'/digest
    folder.mkdir(parents=True,exist_ok=True)
    target = folder/('physx_ray_batch'+sysconfig.get_config_var('EXT_SUFFIX'))
    if not target.exists():
        command = ['c++','-O3','-shared','-std=c++17','-fPIC','-fabi-version=16',
                   '-DPYBIND11_INTERNALS_VERSION=4',*[f'-I{p}' for p in includes],
                   str(source),'-o',str(target)]
        subprocess.run(command,check=True,capture_output=True,text=True,timeout=60.)
    spec = importlib.util.spec_from_file_location('physx_ray_batch',target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeRaycaster:
    def __init__(self, query, directions, robot_scope, output, threads=4):
        self.native = load_native(output)
        self.query = query
        self.directions = np.asarray(directions,dtype=np.float32)
        self.robot_scope = robot_scope
        self.classes = {}
        self.validated_rays = 0
        self.threads = threads
        self.batch = self.native.RayBatch(threads)
        self.timings=[]
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='physx-ray-dispatch')
        self.executor.submit(lambda:None).result()
        self.pending=None

    def begin(self,origins):
        if self.pending is not None:raise RuntimeError('ray query already in flight')
        self.pending=self.executor.submit(self.scan,np.asarray(origins,dtype=np.float32).copy())

    def finish(self):
        if self.pending is None:raise RuntimeError('no ray query in flight')
        try:return self.pending.result()
        finally:self.pending=None

    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=True)
        self.batch.close()

    def scan(self, origins, validate=False):
        from pxr import PhysicsSchemaTools
        from carb import Float3
        origins = np.asarray(origins,dtype=np.float32)
        started=time.monotonic()
        if not np.isfinite(origins).all():raise ValueError('nonfinite LiDAR origins')
        hit, positions, identities, labels, pairs = self.batch.scan(self.query,origins,self.directions)
        native_done=time.monotonic()
        if validate:
            for k,(present,point,pair) in enumerate(zip(hit,positions,identities)):
                origin = Float3(*map(float,origins[k//len(self.directions)]))
                direction = Float3(*map(float,self.directions[k%len(self.directions)]))
                old = self.query.raycast_closest(origin,direction,4.5)
                if bool(present) != old['hit']:raise RuntimeError(f'native hit mismatch at ray {k}')
                if present:
                    np.testing.assert_array_equal(point,tuple(old['position']))
                    paths = tuple(str(PhysicsSchemaTools.intToSdfPath(int(v))) for v in pair)
                    if paths != (old['collision'],old['rigidBody']):
                        raise RuntimeError(f'native identity mismatch at ray {k}')
            self.validated_rays += len(hit)
        classes = np.zeros((len(pairs),3),dtype=bool)
        paths_list = []
        for k,pair in enumerate(pairs):
            if k==0:
                paths_list.append(None)
                continue
            key = tuple(pair)
            cached = self.classes.get(key)
            if cached is None:
                paths = tuple(str(PhysicsSchemaTools.intToSdfPath(int(v))) for v in pair)
                flags = (any(p.startswith(self.robot_scope+'/') for p in paths),
                         any(p.startswith('/World/Moving/') for p in paths),
                         'BlueSubject' in paths[0])
                cached = self.classes[key] = (paths,flags)
            paths_list.append(cached[0]);classes[k] = cached[1]
        flags = classes[labels]
        audit = None
        if validate:
            audit=[]
            for k,paths in enumerate(paths_list):
                if k==0:continue
                selected = np.flatnonzero(labels==k)
                audit.append(dict(collision=paths[0],rigid_body=paths[1],count=len(selected),
                                  sample_points=positions[selected[:3]].tolist()))
        n = len(self.directions)
        if not validate:
            self.timings.append(dict(query_s=native_done-started,classification_s=time.monotonic()-native_done))
        return dict(positions=positions.reshape(len(origins),n,3),
                    hit=hit.reshape(len(origins),n),flags=flags.reshape(len(origins),n,3),
                    audit=audit)
