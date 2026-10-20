"""One episode contract, two existing closed-loop executors.

PhysX runs in an isolated interpreter using the repository's bootstrap, bounded
asynchronous controller and bounded recovery watchdog. No physics pause awaits an Agent.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .episode import Episode, write_json, source_fingerprints

PACKAGE = Path(__file__).resolve().parents[1]
OFFICIAL_API = 'https://api.typesafe.ai/v1/systemone'


def execution_environment(backend, base=None):
    """Keep system ROS available to xacro while bootstrap isolates Isaac's ABI."""
    environment = dict(os.environ if base is None else base)
    paths = [str(PACKAGE)]
    if backend == 'physx':
        ros = Path(environment.get('CINEGYM_ROS_PREFIX', '/opt/ros/'+environment.get('ROS_DISTRO', 'humble')))
        for pattern in ('local/lib/python*/dist-packages', 'lib/python*/site-packages'):
            paths.extend(str(p) for p in sorted(ros.glob(pattern)) if p.is_dir())
        if (ros/'bin').is_dir():
            environment['PATH'] = str(ros/'bin')+os.pathsep+environment.get('PATH', os.defpath)
            environment['AMENT_PREFIX_PATH'] = str(ros)+os.pathsep+environment.get('AMENT_PREFIX_PATH', '')
        numeric = environment.get('JEV_PHYSX_NUMERIC_SITE')
        bundled = PACKAGE/'results/physx_cpu_parity_20261016/physx_numeric'
        if not numeric and (bundled/'osqp').is_dir():
            environment['JEV_PHYSX_NUMERIC_SITE'] = str(bundled)
    paths.extend(environment.get('PYTHONPATH', '').split(os.pathsep))
    environment['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(p for p in paths if p))
    environment.setdefault('OPENBLAS_NUM_THREADS', '1')
    environment.setdefault('OMP_NUM_THREADS', '1')
    environment['PYTHONUNBUFFERED'] = '1'
    key = PACKAGE.parents[1]/'typesafe_Jev_api_key'
    if key.is_file():
        environment.setdefault('TYPESAFE_API_KEY_FILE', str(key))
    return environment


def build_command(case, output, *, backend='cpu', mode='jev', python=None,
                  isaac_python=None, api_url=None, model=None, extra_args=()):
    case, output = Path(case).resolve(), Path(output).resolve()
    Episode.load(case)  # verify exact paired inputs before starting a backend
    if backend not in ('cpu', 'physx') or mode not in ('jev', 'strict', 'rule'):
        raise ValueError('backend must be cpu/physx and mode jev/strict/rule')
    extra_args = list(extra_args)
    # These values define experiment identity and timing; they cannot be
    # overridden through a second copy of an argparse option.
    protected = {'--case', '--scenarios', '--output', '--mode', '--pace', '--physics-hz', '--camera-output'}
    if any(any(name.startswith(arg.split('=', 1)[0]) for name in protected)
           for arg in extra_args if arg.startswith('--')):
        raise ValueError('extra_args may not override case, output, mode, pace or physical defaults')
    common = ['--scenarios', str(case.parent), '--case', case.name, '--output', str(output),
              '--mode', mode, '--pace', '1']
    if mode == 'jev':
        api_url = api_url or OFFICIAL_API
        model = model or 'jev-1.14.0'
    if api_url is not None:
        common += ['--api-url', api_url]
    if model is not None:
        common += ['--model', model]
    python = str(python or sys.executable)
    if backend == 'cpu':
        return [python, '-m', 'jev_agent_wbc.runner', *common, *extra_args]
    isaac_python = isaac_python or os.environ.get('CINEGYM_ISAAC_PYTHON')
    if not isaac_python or not Path(isaac_python).expanduser().is_file():
        raise ValueError('PhysX requires --isaac-python or CINEGYM_ISAAC_PYTHON pointing to Isaac Sim Python')
    return [python, str(PACKAGE/'scripts/jvs_isaac_bootstrap.py'), '--isaac-python',
        str(Path(isaac_python).expanduser().absolute()), '--runner', str(PACKAGE/'scripts/run_isaac.py'), '--',
        *common, '--physics-hz', '100', '--camera-output', 'execution',
        '--no-startup-preplan', *extra_args]


@dataclass
class RunHandle:
    process: subprocess.Popen
    output: Path
    result_folder: Path
    log: object
    started: float

    def poll(self):
        """Nonblocking status. Progress never suspends the simulator."""
        code = self.process.poll()
        result = dict(running=code is None, returncode=code, output=str(self.output),
                      wall_s=time.monotonic()-self.started)
        path = self.result_folder/'summary.json'
        if code is not None:
            self.log.close()
            if path.is_file():
                result['summary'] = json.loads(path.read_text())
        return result

    def wait(self, timeout=None):
        self.process.wait(timeout=timeout)
        return self.poll()

    def close(self):
        if self.process.poll() is None:
            # Give runner finally blocks a chance to close isolated controller
            # process groups and Isaac, before escalating to a bounded kill.
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def start_closed_loop(case, output, **options):
    """Start CPU or PhysX. The existing Agent consumes observations in its loop."""
    command = build_command(case, output, **options)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = execution_environment(options.get('backend', 'cpu'))
    episode = Episode.load(case)
    write_json(output/'cinegym_run.json', dict(backend=options.get('backend', 'cpu'),mode=options.get('mode','jev'),
        episode_sha256=episode.metadata['sha256'], seed=episode.metadata['seed'],
        family=episode.metadata['family'], difficulty=episode.metadata['difficulty'],
        case=str(Path(case).resolve()), command=command, pace=1., source_sha256=source_fingerprints(),
        physics_hz=100. if options.get('backend') == 'physx' else None))
    log = (output/'backend.log').open('w')
    try:
        process = subprocess.Popen(command, cwd=PACKAGE, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        log.close()
        raise
    return RunHandle(process, output, output/f'{Path(case).name}_{options.get("mode", "jev")}', log, time.monotonic())


def run_closed_loop(case, output, *, timeout=None, **options):
    with start_closed_loop(case, output, **options) as handle:
        try:
            return handle.wait(timeout)
        except subprocess.TimeoutExpired:
            handle.close()
            result = handle.poll()
            result['timed_out'] = True
            write_json(handle.output/'cinegym_timeout.json', result)
            return result
