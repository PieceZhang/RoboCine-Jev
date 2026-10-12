"""Independent swept clearance audit from Isaac rigid-body poses at every step."""
import csv
import gzip
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


class CollisionAudit:
    def __init__(self,folder,collision):
        self.collision=collision
        self.file=gzip.open(Path(folder)/'collision.csv.gz','wt',newline='')
        self.writer=csv.writer(self.file)
        self.writer.writerow(['time_s','minimum_clearance_m','continuous_lower_m',
                              *[f'body{b}_clearance_m' for b in range(8)]])
        self.previous=None
        self.minimum=np.inf;self.lower=np.inf;self.collisions=0;self.unsafe=0;self.count=0
        self.max_bound=0.
        # Radii about link origins cover each body's local geometry.
        self.radii=np.array([.39,.11,.231,.213,.118,.098,.145,.056])

    def record(self,t,frames):
        c=self.collision
        values=np.array([c.geometry.body_sdf(c.centers,frames,b)-c.radii for b in range(8)])
        d=np.min(values,axis=1)
        bound=np.zeros(8)
        if self.previous is not None:
            old_frames,old_d=self.previous
            for b,((p,r),(op,orr)) in enumerate(zip(frames,old_frames)):
                angle=Rotation.from_matrix(r@orr.T).magnitude()
                bound[b]=np.linalg.norm(p-op)+self.radii[b]*angle
            lower=np.minimum(d,old_d)-bound/2
        else:
            lower=d
        self.previous=([(p.copy(),r.copy()) for p,r in frames],d)
        minimum=float(np.min(d));lb=float(np.min(lower))
        self.writer.writerow([t,minimum,lb,*d])
        self.minimum=min(self.minimum,minimum);self.lower=min(self.lower,lb)
        self.max_bound=max(self.max_bound,float(np.max(bound/2)))
        self.collisions+=int(minimum<0);self.unsafe+=int(lb<c.margin);self.count+=1

    def finish(self):
        self.file.close()
        return dict(samples=self.count,minimum_clearance_m=self.minimum,continuous_lower_m=self.lower,
                    collision_sample_rate=self.collisions/max(self.count,1),
                    unsafe_sample_rate=self.unsafe/max(self.count,1),minimum_required_m=self.collision.margin,
                    max_interpolation_bound_m=self.max_bound,
                    collision_free=self.lower>=0,clearance_pass=self.lower>=self.collision.margin,
                    scope='every PhysX step; rigid translation/SLERP sweep between adjacent poses; conservative URDF geometry')
