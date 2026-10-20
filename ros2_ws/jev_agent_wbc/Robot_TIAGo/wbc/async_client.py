"""Nonblocking physical-loop client with finite freshness and recovery budgets."""
import pickle
from pathlib import Path
import socket
import subprocess
import time

import numpy as np


class AsyncGiskardClient:
    def __init__(self,metadata,log_path,python=None,robot_dir=None,module_dir=None,spec_path=None,backend='giskard-core',mesh_cache=None,sensor_config=None):
        self.root = Path(robot_dir) if robot_dir else Path(__file__).resolve().parents[1]
        self.module_dir=Path(module_dir) if module_dir else self.root/'wbc'
        self.spec_path=Path(spec_path) if spec_path else self.root/'robot.json'
        self.backend=backend
        self.mesh_cache=Path(mesh_cache) if mesh_cache else self.root/'wbc/cache/world_binary_mesh'
        self.sensor_config=Path(sensor_config) if sensor_config else self.root/'sensors.json'
        self.initialization=None
        self.python = str(python or self.root/'wbc/.venv/bin/python')
        self.metadata = metadata
        self.log = Path(log_path).open('a')
        self.restarts,self.sequence,self.pending = 0,0,None
        self.last,self.ready = None,False
        self.events=[]
        self.closed=False
        self._start()

    def _start(self):
        self.channel,child = socket.socketpair(socket.AF_UNIX,socket.SOCK_SEQPACKET)
        self.channel.setblocking(False)
        self.process = subprocess.Popen([self.python,str(self.module_dir/'worker.py'),'--fd',str(child.fileno()),'--metadata',str(self.metadata),
            '--robot-dir',str(self.root),'--spec',str(self.spec_path),'--backend',self.backend,'--mesh-cache',str(self.mesh_cache),'--sensor-config',str(self.sensor_config)],
            pass_fds=(child.fileno(),),stdout=self.log,stderr=self.log)
        child.close()
        self.ready,self.pending,self.last = False,None,None
        self.exit_reported=False

    def prewarm(self,timeout_s=60.):
        deadline = time.monotonic()+timeout_s
        while time.monotonic()<deadline:
            self.poll(0.)
            if self.ready:
                return
            if self.process.poll() is not None:
                raise RuntimeError('Giskard prewarm failed; see worker log')
            time.sleep(.005)  # before the first physical advance only
        raise RuntimeError('Giskard prewarm timeout before physical execution')

    def poll(self,sim_s):
        while True:
            try:
                data = self.channel.recv(512*1024)
            except BlockingIOError:
                break
            except ConnectionResetError:
                self.events.append(dict(sim_s=sim_s,event='bounded_channel_reset'))
                self.ready=False;self.last=None
                self.process.terminate()
                break
            if not data:
                break
            packet = pickle.loads(data)
            if packet['kind']=='ready':
                self.ready=True
                self.initialization=packet
            elif packet['kind']=='command':
                self.pending=None
                self.last=packet
                if not packet['ok']:
                    self.events.append(dict(sim_s=sim_s,event='solver_rejected',error=packet['error']))
        if self.process.poll() is not None:
            if not self.exit_reported:
                self.events.append(dict(sim_s=sim_s,event='worker_exited',code=self.process.returncode))
                self.exit_reported=True
            if self.restarts<2:
                self.restarts+=1
                self.channel.close();self._start()
            else:
                self.ready=False;self.last=None;self.pending=None
        if self.pending is not None and sim_s-self.pending['sample_s']>.10:
            self.events.append(dict(sim_s=sim_s,event='worker_deadline',sequence=self.pending['sequence']))
            self.process.terminate()
            self.ready=False

    def command(self,request):
        self.poll(request['sample_s'])
        packet=self.last
        valid = (packet is not None and packet['ok'] and packet['phase']==request['phase']
            and 0.<=request['sample_s']-packet['sample_s']<=.04)
        rates = np.asarray(packet['command']).copy() if valid else np.zeros(10)
        status = dict(fresh_native_command=valid,command_sample_s=packet['sample_s'] if packet else None,
            sample_age_s=request['sample_s']-packet['sample_s'] if packet else None,
            worker_ready=self.ready,restarts=self.restarts,
            error=packet.get('error') if packet else None,
            collision=packet.get('collision') if packet else None,
            worker_wall_s=packet.get('worker_wall_s') if packet else None,
            hard_constraint_violation=packet.get('hard_constraint_violation') if packet else None)
        if packet:
            status.update(full_world_executor=packet.get('full_world_executor',False),
                framework_tick_wall_s=packet.get('framework_tick_wall_s'),projection_norm=packet.get('projection_norm'),
                upstream_proposal=packet.get('upstream_proposal'),upstream_derivative_mode=packet.get('upstream_derivative_mode'))
            status['measured_native_camera_consistency']=packet.get('measured_native_camera_consistency')
        if self.ready and self.pending is None:
            self.sequence+=1
            payload=dict(request,sequence=self.sequence)
            try:
                self.channel.send(pickle.dumps(payload,protocol=5))
                self.pending=payload
            except BlockingIOError:
                self.events.append(dict(sim_s=request['sample_s'],event='bounded_channel_busy'))
        return rates,status

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            self.channel.send(pickle.dumps(dict(kind='shutdown')))
        except (OSError,BlockingIOError):
            pass
        try:
            self.process.wait(timeout=2.)
        except subprocess.TimeoutExpired:
            self.process.terminate()
        self.channel.close();self.log.close()
