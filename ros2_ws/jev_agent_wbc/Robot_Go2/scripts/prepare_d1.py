#!/usr/bin/env python3
"""Assemble the two upstream URDFs without changing the vendored originals."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor/Isaac-sim-go2-with-arm"
OUT = ROOT / "assets/generated"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    go2 = ET.parse(SOURCE / "urdf/go2_description.urdf").getroot()
    arm = ET.parse(SOURCE / "urdf/d1_description.urdf").getroot()
    go2.set("name", "go2_d1")
    for element in list(arm):
        go2.append(element)
    joint = ET.SubElement(go2, "joint", name="go2_to_d1_arm", type="fixed")
    ET.SubElement(joint, "parent", link="base")
    ET.SubElement(joint, "child", link="base_link")
    ET.SubElement(joint, "origin", xyz="0.05 0 0.10", rpy="0 0 0")
    for mesh in go2.iter("mesh"):
        mesh_path = SOURCE / "meshes" / Path(mesh.get("filename")).name
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        mesh.set("filename", str(mesh_path))
    # D1 has six revolute joints and two prismatic fingers. The upstream
    # controller commands only Joint7_1. Preserve both fingers and the URDF.
    ET.indent(go2)
    output = OUT / "go2_d1.urdf"
    ET.ElementTree(go2).write(output, encoding="utf-8", xml_declaration=True)
    report = {
        "output": str(output), "mount_xyz_m": [0.05, 0, 0.10],
        "upstream_separate_urdfs": True,
        "actuated_joints": [{"name": j.get("name"), "type": j.get("type")}
                            for j in go2.findall("joint") if j.get("type") != "fixed"],
        "mesh_paths_resolved": True,
        "note": "D1 is 6 arm revolute DOF + two prismatic fingers; not interchangeable with Piper policies. Mount is a local integration choice.",
    }
    (OUT / "go2_d1_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
