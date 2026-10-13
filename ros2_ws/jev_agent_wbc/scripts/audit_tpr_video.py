#!/usr/bin/env python3
"""Check the visible orange sphere against measured world poses, frame by frame.

For this fixed-view demo only: use RGB pixels, not requested obstacle commands,
to detect a frozen or incorrectly projected visual despite moving telemetry.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def project(position, calibration):
    local=np.asarray(calibration['rotation']).T@(np.asarray(position)-calibration['position'])
    if local[0]<=0:
        raise ValueError('obstacle behind overview camera')
    focal=.5*calibration['height']/np.tan(np.deg2rad(calibration['vfov_deg']/2))
    return np.array([calibration['width']/2-focal*local[1]/local[0],
                     calibration['height']/2-focal*local[2]/local[0]]),focal*.07/local[0]


def audit(run):
    run=Path(run)
    calibration=json.loads((run/'video/overview_calibration.json').read_text())
    metadata=json.loads((run/'video/frames.json').read_text())
    rows=json.loads((run/'telemetry.json').read_text())
    times=np.array([row['t'] for row in rows])
    records=[]
    for frame in metadata:
        row=rows[int(np.argmin(abs(times-frame['t'])))]
        position=row.get('obstacle_physx_position',row['obstacle'])
        expected,radius=project(position,calibration)
        path=run/f'video/overview/{frame["frame"]:05d}.jpg'
        rgb=cv2.imread(str(path))
        if rgb is None:
            raise RuntimeError(f'missing video frame: {path}')
        hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,(0,120,100),(18,255,255))
        count,_,stats,centroids=cv2.connectedComponentsWithStats(mask)
        area=np.pi*radius**2
        candidates=[i for i in range(1,count)
            if .2*area<stats[i,4]<2.*area and .5<stats[i,2]/max(stats[i,3],1)<2.]
        best=min(candidates,key=lambda i:np.linalg.norm(centroids[i]-expected)) if candidates else None
        detected=None if best is None else centroids[best]
        error=None if detected is None else float(np.linalg.norm(detected-expected))
        records.append(dict(frame=frame['frame'],t=frame['t'],expected_pixel=expected.tolist(),
            detected_pixel=None if detected is None else detected.tolist(),error_px=error))
    found=[r for r in records if r['detected_pixel'] is not None]
    errors=np.array([r['error_px'] for r in found])
    expected_span=float(np.linalg.norm(np.ptp([r['expected_pixel'] for r in records],axis=0)))
    detected_span=float(np.linalg.norm(np.ptp([r['detected_pixel'] for r in found],axis=0))) if found else 0.
    passed=bool(len(found)>=.95*len(records) and len(errors) and errors.max()<5.
                and (expected_span<10. or detected_span>.75*expected_span))
    result=dict(pass_motion=passed,frames=len(records),detected_frames=len(found),
        expected_span_px=expected_span,detected_span_px=detected_span,
        position_error_max_px=None if not len(errors) else float(errors.max()),
        position_error_p95_px=None if not len(errors) else float(np.percentile(errors,95)),
        scope='orange-sphere RGB centroid versus fixed-camera projection; visibility and motion only',
        pose_source='PhysX read-back' if 'obstacle_physx_position' in rows[0] else 'legacy trajectory log (no physical read-back)',
        frames_checked=records)
    (run/'video_motion_audit.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    result=audit(parser.parse_args().run)
    print(json.dumps({k:v for k,v in result.items() if k!='frames_checked'},indent=2))
    return 0 if result['pass_motion'] else 2


if __name__=='__main__':
    raise SystemExit(main())
