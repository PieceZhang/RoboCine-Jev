"""Seeded episodes and balanced, reproducible benchmark suites."""
from collections import Counter
from dataclasses import asdict, replace
from contextlib import redirect_stdout
import io
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from jev_agent_wbc.nominal import nominal_rollout
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.whole_body_qp import whole_body_kinematics
from .config import DIFFICULTIES, FAMILIES, GymConfig
from .episode import Episode, fingerprint, write_json, source_fingerprints
from .paths import camera_task
from .obstacles import fit_obstacles
from .authoring import WithTerminalHold, audit_nominal

VERSION = 'cinegym/v3'


def generate_episode(seed, config=None):
    config = config or GymConfig()
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    split = ('train', 'validation', 'test').index(config.split)
    rng = np.random.default_rng(np.random.SeedSequence([seed, split, 20261019]))
    family = str(rng.choice(FAMILIES)) if config.family == 'mixed' else config.family
    rejected = []
    for attempt in range(config.max_attempts):
        task, scene, parameters = camera_task(rng, family, config)
        reference = OriginalReference(task, scene['subject']['position'])
        # Some OSQP builds print a polishing notice at every unconstrained step.
        with redirect_stdout(io.StringIO()):
            rollout_reference=WithTerminalHold(reference) if config.generation_style=='natural' else reference
            ts, xs, failure = nominal_rollout(rollout_reference, scene['initial_state'], scene['base_z'])
        if failure or ts[-1] < reference.duration-1e-8:
            rejected.append('nominal controller failed: '+failure)
            continue
        errors, angles = [], []
        indices = np.unique(np.r_[np.arange(0, len(ts), 5), len(ts)-1])
        for t, x in zip(ts[indices], xs[indices]):
            p, r, _ = whole_body_kinematics(x, scene['base_z'])
            target, rotation, _ = reference.sample(t)
            errors.append(float(np.linalg.norm(p-target)))
            angles.append(float(Rotation.from_matrix(r@rotation.T).magnitude()))
        if max(errors) > .045 or max(angles) > .08:
            rejected.append(f'nominal tracking exceeds authoring bounds: {max(errors):.4f} m, {max(angles):.4f} rad')
            continue
        nominal_audit=audit_nominal(reference,scene,ts,xs) if config.generation_style=='natural' else None
        if nominal_audit is not None and not nominal_audit['accepted']:
            rejected.append('nominal shot acceptance: '+','.join(nominal_audit['failed_checks']))
            continue
        times = np.linspace(0., rollout_reference.duration, int(np.ceil(rollout_reference.duration/config.audit_step_s))+1)
        states = np.column_stack([np.interp(times, ts, xs[:, i]) for i in range(9)])
        pool=None
        mixed_boxes=config.ground_boxes and config.generation_style=='natural'
        if config.require_avoidance_witness or mixed_boxes:
            from .solvability import timing_pool
            pool=timing_pool(reference,scene,ts,xs)
            if not pool:
                rejected.append('restricted timing family has no actuator/shot-feasible witness')
                continue
        obstacles=None
        for obstacle_attempt in range(4 if pool is not None else 1):
            try:
                obstacles,audit=fit_obstacles(rng,scene,times,states,config,pool,task['shot_phase'])
                if mixed_boxes:
                    from .ground_boxes import add_ground_boxes
                    obstacles,audit=add_ground_boxes(rng,scene,times,states,obstacles,audit,pool,config)
                break
            except ValueError as error:
                obstacles=None
                rejected.append(str(error))
        if obstacles is None:
            continue
        scene['obstacles'] = obstacles
        solvability=dict(status='unverified',scope='Nominal feasibility does not prove avoidance solvability')
        if pool is not None:
            from .solvability import finish_witness
            from scipy.interpolate import CubicHermiteSpline
            curves=[CubicHermiteSpline([k['t'] for k in o['keyframes']],
                [k['position'] for k in o['keyframes']],[k['velocity'] for k in o['keyframes']]) for o in obstacles]
            solvability=finish_witness(reference,scene,pool,audit['timing_witness_indices'],curves)
            if solvability is None:
                rejected.append('safe timing witnesses fail obstacle-aware shot acceptance')
                continue
        scene['name'] = f'{config.split}_{seed:08d}_{family}_{config.difficulty}'
        metadata = dict(version=VERSION, seed=seed, split=config.split, family=family,
            difficulty=config.difficulty, config=asdict(config), parameters=parameters,
            duration_s=reference.duration, obstacle_count=len(obstacles), accepted_attempt=attempt,
            rejected_attempts=rejected, nominal_tracking_max_m=max(errors),
            nominal_orientation_max_rad=max(angles), audit=audit,
            nominal_shot_acceptance=nominal_audit,
            avoidance_solvability=solvability,
            source_designs='scenarios.py six primitives and scenarios/combo_01..06 action sequences',
            source_sha256=source_fingerprints(),
            sha256=fingerprint(task, scene))
        return Episode(task, scene, metadata)
    raise RuntimeError(f'failed to generate seed={seed}, family={family}: {rejected}')


def generate_suite(output, count=42, seed=0, config=None, balanced=True):
    if type(count) is not int or count <= 0 or type(seed) is not int or seed < 0:
        raise ValueError('count must be positive and seed nonnegative integers')
    config = config or GymConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for i in range(count):
        selected = config
        if balanced:
            family_count = len(FAMILIES) if config.family == 'mixed' else 1
            selected = replace(config, family=FAMILIES[i % len(FAMILIES)] if config.family == 'mixed' else config.family,
                               difficulty=DIFFICULTIES[(i//family_count) % 3])
        episode = generate_episode(seed+i, selected)
        episode.save(output/episode.scene['name'])
        records.append(dict(case=episode.scene['name'], seed=seed+i, family=episode.metadata['family'],
                            difficulty=selected.difficulty, sha256=episode.metadata['sha256']))
        # A partial suite remains auditable if generation is interrupted.
        write_json(output/'catalog.json', dict(version=VERSION, seed=seed, split=config.split,
            requested_count=count, complete=len(records) == count, cases=[r['case'] for r in records], episodes=records,
            coverage=dict(families=dict(Counter(r['family'] for r in records)),
                          difficulties=dict(Counter(r['difficulty'] for r in records)))))
        print(f'generated {len(records)}/{count}: {episode.scene["name"]}', flush=True)
    return records
