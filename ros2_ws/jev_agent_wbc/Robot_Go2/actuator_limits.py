"""Read explicit torque caps from the pinned upstream Go2-Piper MuJoCo model.

The XML motors have unit gear, so symmetric ctrlrange is torque in Nm. This
does not infer unspecified limits or borrow D1/another arm's hardware values.
"""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def load_motor_limits(path):
    path=Path(path);root=ET.parse(path).getroot();result={}
    for motor in root.findall('./actuator/motor'):
        joint=motor.attrib['joint'];bounds=np.fromstring(motor.attrib['ctrlrange'],sep=' ')
        gear=np.fromstring(motor.attrib.get('gear','1'),sep=' ')
        if (joint in result or bounds.shape!=(2,) or not np.isfinite(bounds).all()
                or not bounds[1]>0 or not np.isclose(bounds[0],-bounds[1])
                or gear.shape!=(1,) or not np.isclose(gear[0],1.)):
            raise ValueError('unique explicit symmetric unit-gear motor limits required')
        result[joint]=float(bounds[1])
    if len(result)!=18:
        raise ValueError('exactly the 12 leg and 6 Piper motors required')
    return dict(effort_limits_nm=result,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                semantics='explicit upstream Go2-Piper unit-gear motor ctrlrange, not simulation default maxForce')
