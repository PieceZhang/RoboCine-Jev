#!/usr/bin/env python3
"""Read-only probe of a future Agent task against one installed WBC adapter.

This does not dispatch motion. Exit 0: task supported by the declared live
adapter; 2: unavailable/unsupported/stale; malformed messages raise an error.
Native numerical feasibility and collision validation remain required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from jev_agent_wbc.representation import Observation
from jev_agent_wbc.wbc_evidence import inspect_evidence
from jev_agent_wbc.wbc_protocol import Capabilities, TimedTask, negotiate


def probe(task_document, observation_document, readiness, now_s, *, robot_dir=None):
    task = TimedTask.from_dict(task_document)
    observation = Observation.from_dict(observation_document)
    # An installed numerical framework alone is not a live Representation ingress.
    evidence_errors = (inspect_evidence(robot_dir, readiness)[0] if robot_dir is not None
                       else ['evidence_location_required'])
    ready = (not evidence_errors and readiness.get('backend_runtime_ready') is True
             and readiness.get('representation_ingress_verified') is True)
    capability = Capabilities(
        str(readiness.get('recommended_backend', readiness.get('backend', 'unconfigured'))),
        ready, frozenset(readiness.get('protocol_features', [])))
    result = negotiate(task, observation, capability, now_s)
    result['readiness_errors'] = evidence_errors
    result.update(probe_only=True, motion_dispatched=False, avoidance_verified=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-dir', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--observation', type=Path, required=True)
    parser.add_argument('--now-s', type=float, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = args.robot_dir/'WBC_READINESS.json'
    result = probe(json.loads(args.task.read_text()), json.loads(args.observation.read_text()),
                   json.loads(manifest.read_text()), args.now_s, robot_dir=args.robot_dir)
    result['inputs_sha256'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (args.task, args.observation, manifest)}
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(payload)
    print(payload, end='')
    return 0 if result['supported'] else 2


if __name__ == '__main__':
    sys.exit(main())
