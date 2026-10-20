#!/usr/bin/env python3
"""Read-only authored USD/URDF inertias, with fixed-link aggregation in body axes."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics


def rotation_rpy(rpy):
    r,p,y=rpy;cr,sr=np.cos(r),np.sin(r);cp,sp=np.cos(p),np.sin(p);cy,sy=np.cos(y),np.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
        [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def origin(element):
    result=np.eye(4)
    if element is not None:
        result[:3,3]=np.fromstring(element.get('xyz','0 0 0'),sep=' ')
        result[:3,:3]=rotation_rpy(np.fromstring(element.get('rpy','0 0 0'),sep=' '))
    return result


def quat_rotation(q):
    w=float(q.GetReal());x,y,z=map(float,q.GetImaginary())
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def parallel(c):return np.dot(c,c)*np.eye(3)-np.outer(c,c)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,required=True);parser.add_argument('--urdf',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();stage=Usd.Stage.Open(str(args.stage));meters=UsdGeom.GetStageMetersPerUnit(stage)
    kilograms=UsdPhysics.GetStageKilogramsPerUnit(stage)
    calibration=json.loads(args.calibration.read_text())['physical_body_frame_offsets']
    bodies={}
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot'),Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):continue
        api=UsdPhysics.MassAPI(prim);mass=api.GetMassAttr().Get();com=api.GetCenterOfMassAttr().Get()
        diagonal=api.GetDiagonalInertiaAttr().Get();axes=api.GetPrincipalAxesAttr().Get()
        defined=bool(mass is not None and mass>0 and com is not None and diagonal is not None
            and np.isfinite([*com,*diagonal]).all() and min(diagonal)>0)
        if not defined:raise ValueError('Authored mass/inertia unresolved: '+str(prim.GetPath()))
        R=quat_rotation(axes);I=R@np.diag(np.array(diagonal)*kilograms*meters**2)@R.T
        bodies[prim.GetName()]=dict(path=str(prim.GetPath()),mass_kg=float(mass*kilograms),
            center_body_m=(np.array(com)*meters).tolist(),inertia_body_kgm2=I.tolist(),
            principal_axes_wxyz=[float(axes.GetReal()),*map(float,axes.GetImaginary())],
            diagonal_inertia_kgm2=(np.array(diagonal)*kilograms*meters**2).tolist())
    root=ET.parse(args.urdf).getroot();links={link.get('name'):link for link in root.findall('link')}
    parents={j.find('child').get('link'):(j.find('parent').get('link'),origin(j.find('origin')),j.get('type'))
        for j in root.findall('joint')}
    groups={name:[] for name in bodies}
    for name,link in links.items():
        inertial=link.find('inertial')
        if inertial is None:continue
        body=name;T=np.eye(4)
        while body not in bodies:
            parent,placement,kind=parents[body]
            if kind!='fixed':raise ValueError('Nonfixed URDF body missing from USD: '+body)
            T=placement@T;body=parent
        frame=T@origin(inertial.find('origin'));mass=float(inertial.find('mass').get('value'))
        a=inertial.find('inertia').attrib
        I=np.array([[float(a['ixx']),float(a['ixy']),float(a['ixz'])],
            [float(a['ixy']),float(a['iyy']),float(a['iyz'])],
            [float(a['ixz']),float(a['iyz']),float(a['izz'])]])
        groups[body].append(dict(link=name,mass_kg=mass,com=frame[:3,3],inertia=frame[:3,:3]@I@frame[:3,:3].T))
    comparisons={}
    for name,native in bodies.items():
        contributors=groups[name];mass=sum(c['mass_kg'] for c in contributors)
        com=sum((c['mass_kg']*c['com'] for c in contributors),np.zeros(3))/mass
        inertia=sum((c['inertia']+c['mass_kg']*parallel(c['com']) for c in contributors),np.zeros((3,3)))-mass*parallel(com)
        report=dict(usd=native,urdf_aggregated_mass_kg=mass,urdf_aggregated_com_m=com.tolist(),
            urdf_aggregated_inertia_kgm2=inertia.tolist(),mass_delta_usd_minus_urdf_kg=native['mass_kg']-mass,
            fixed_contributors=[dict(link=c['link'],mass_kg=c['mass_kg']) for c in contributors],
            calibrated_frame_available=name in calibration)
        if name in calibration:
            T=np.array(calibration[name]);native_com=T[:3,:3]@native['center_body_m']+T[:3,3]
            native_I=T[:3,:3]@native['inertia_body_kgm2']@T[:3,:3].T
            report.update(com_difference_m=float(np.linalg.norm(native_com-com)),
                inertia_frobenius_difference_kgm2=float(np.linalg.norm(native_I-inertia)))
        comparisons[name]=report
    result=dict(schema='g1_authored_inertia_audit/v1',physical_execution=False,
        scope='Authored native USD vs official URDF with exact fixed-link mass aggregation. No physical parameter or model is changed. PhysX view values still require independent runtime verification.',
        stage_meters_per_unit=meters,stage_kilograms_per_unit=kilograms,rigid_body_count=len(bodies),
        authored_usd_total_mass_kg=sum(x['mass_kg'] for x in bodies.values()),
        urdf_total_mass_kg=sum(c['mass_kg'] for group in groups.values() for c in group),bodies=comparisons,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),args.stage,args.urdf,args.calibration]})
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('total USD/URDF',result['authored_usd_total_mass_kg'],result['urdf_total_mass_kg'])
    for name,item in comparisons.items():
        if abs(item['mass_delta_usd_minus_urdf_kg'])>1e-4:
            print(name,'mass delta',item['mass_delta_usd_minus_urdf_kg'],'COM delta',item.get('com_difference_m'))


if __name__=='__main__':main()
