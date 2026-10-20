"""Read-only full-window G1 joint ranking, for diagnosis rather than acceptance."""
import hashlib
import json
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
RUNS = ('adaptive_recovery_sameprofile_02', 'adaptive_ground_sameprofile_01',
        'adaptive_pink_soft_profile_01', 'adaptive_yaw_gain2_01', 'default_upper_yaw_01')


def main():
    records, sources = {}, {}
    for run in RUNS:
        source = PACKAGE/'Robot_G1/results'/run/'measurements.jsonl'
        sources[str(source.relative_to(PACKAGE))] = hashlib.sha256(source.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        records[run] = {}
        for name in dict.fromkeys(r['case'] for r in rows if r['case'] != 'settle'):
            stopped = [r for r in rows if r['case'] == name and r['phase'] == 'stop']
            stamps = np.asarray([r['stamp_s'] for r in stopped])
            first = max(0, int(np.searchsorted(stamps, stamps[-1]-.5, side='right'))-1)
            tail = stopped[first:]
            span = np.ptp(np.asarray([r['joint_positions'] for r in tail]), axis=0)
            order = np.argsort(span)[::-1]
            records[run][name] = dict(measured_window_s=float(stamps[-1]-stamps[first]),
                joint_order=tail[-1]['joint_names'], joint_span_rad=span.tolist(),
                above_original_10mrad=[tail[-1]['joint_names'][i] for i in order if span[i] >= .01],
                four_largest=[dict(joint=tail[-1]['joint_names'][i], span_rad=float(span[i])) for i in order[:4]])
    output = PACKAGE/'validation/g1_stop_joint_ranking_root_20261020.json'
    with output.open('x') as stream:
        json.dump(dict(schema='g1_stop_joint_diagnostic/v1', diagnostic_only=True,
            scope='Complete original final 0.5s including sample at/before boundary; ranking cannot certify stability.',
            results=records, sources_sha256=sources,
            auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), stream, indent=2)
        stream.write('\n')
    print(output)


if __name__ == '__main__':
    main()
