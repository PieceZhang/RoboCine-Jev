#!/usr/bin/env python3
"""Resolve the pinned PAL TIAGo xacros and copy a portable mesh closure.

Uses the system ROS xacro package without requiring a built ROS workspace.
The original repositories remain unmodified and retain their license files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import math
import copy

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collision_dae_to_stl(original, output):
    """Preserve raw ROS mesh coordinates; reject transforms or other primitives.

    PAL's five collision DAEs have empty image/effect libraries and no node
    transforms. Isaac 5.1's native converter crashes in tinyxml2 on these files.
    Exporting their indexed triangles avoids that native parser path without
    replacing the collision geometry with boxes or changing its scale.
    """
    doc = ET.parse(original)
    nodes = doc.findall('.//{*}visual_scene/{*}node')
    for node in nodes:
        if any(child.tag.split('}')[-1] != 'instance_geometry' for child in node):
            raise ValueError(f'DAE scene transform unsupported: {original}')
    triangles = []
    for mesh in doc.findall('.//{*}geometry/{*}mesh'):
        sources = {}
        for source in mesh.findall('{*}source'):
            array = [float(x) for x in source.find('{*}float_array').text.split()]
            stride = int(source.find('{*}technique_common/{*}accessor').attrib.get('stride', 3))
            sources[source.attrib['id']] = [array[i:i+3] for i in range(0, len(array), stride)]
        vertices = {v.attrib['id']: v.find('{*}input[@semantic="POSITION"]').attrib['source'].lstrip('#')
                    for v in mesh.findall('{*}vertices')}
        for primitive in mesh:
            if primitive.tag.split('}')[-1] in ('source', 'vertices'):
                continue
            if primitive.tag.split('}')[-1] != 'triangles':
                raise ValueError(f'DAE has non-triangle geometry: {original}')
            inputs = primitive.findall('{*}input')
            position = next(item for item in inputs if item.attrib['semantic'] == 'VERTEX')
            stride = max(int(item.attrib.get('offset', 0)) for item in inputs) + 1
            indices = [int(x) for x in primitive.find('{*}p').text.split()]
            offset = int(position.attrib.get('offset', 0))
            points = sources[vertices[position.attrib['source'].lstrip('#')]]
            xyz = [points[indices[i+offset]] for i in range(0, len(indices), stride)]
            triangles.extend([xyz[i:i+3] for i in range(0, len(xyz), 3)])
    with output.open('w') as stream:
        stream.write('solid pal_collision\n')
        for triangle in triangles:
            a, b, c = triangle
            u, v = [b[i]-a[i] for i in range(3)], [c[i]-a[i] for i in range(3)]
            normal = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
            norm = math.sqrt(sum(x*x for x in normal))
            normal = [x/norm for x in normal] if norm else [0.,0.,0.]
            stream.write(' facet normal '+' '.join(map(str, normal))+'\n  outer loop\n')
            for point in triangle:
                stream.write('   vertex '+' '.join(map(str, point))+'\n')
            stream.write('  endloop\n endfacet\n')
        stream.write('endsolid pal_collision\n')
    return len(triangles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', help='download missing repositories at SOURCE_LOCK commits')
    args = parser.parse_args()
    lock = json.loads((ROOT / 'SOURCE_LOCK.json').read_text())
    sources = ROOT / 'assets/sources'
    sources.mkdir(parents=True, exist_ok=True)
    for source in lock['repositories']:
        target = sources / source['name']
        if not target.exists():
            if not args.fetch:
                raise FileNotFoundError(f'{target}; run with --fetch to download official sources')
            subprocess.run(['git', 'init', str(target)], check=True)
            subprocess.run(['git', '-C', str(target), 'remote', 'add', 'origin', source['repository']], check=True)
            subprocess.run(['git', '-C', str(target), 'fetch', '--depth', '1', 'origin', source['commit']], check=True)
            subprocess.run(['git', '-C', str(target), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        commit = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != source['commit']:
            raise ValueError(f'{target} is {commit}, expected {source["commit"]}')
        if subprocess.check_output(['git', '-C', str(target), 'status', '--porcelain'], text=True).strip():
            raise ValueError(f'Official source tree has local modifications: {target}')
    # Resolve $(find package) from the locked package source folders, without
    # injecting local fake ROS packages or relying on host workspace overlays.
    sys.path.insert(0, '/opt/ros/humble/local/lib/python3.10/dist-packages')
    import xacro
    import xacro.substitution_args
    packages = {ET.parse(path).getroot().findtext('name'): path.parent for path in sources.rglob('package.xml')}
    xacro.substitution_args._eval_find = lambda name: str(packages[name])
    entry = packages['tiago_description'] / 'robots/tiago.urdf.xacro'
    expanded = xacro.process_file(str(entry), mappings=lock['configuration']).toxml()
    document = ET.fromstring(expanded)
    removed = {}
    # ROS/Gazebo control tags are not part of the imported mechanical model.
    for tag in ('gazebo', 'transmission', 'ros2_control'):
        removed[tag] = 0
        for element in list(document.findall(tag)):
            document.remove(element)
            removed[tag] += 1
    closure = {}
    for mesh in document.findall('.//mesh'):
        uri = mesh.attrib['filename']
        if not uri.startswith('package://'):
            raise ValueError(f'Unexpected non-package mesh URI: {uri}')
        package, relative = uri[len('package://'):].split('/', 1)
        original = packages[package] / relative
        if not original.is_file():
            raise FileNotFoundError(original)
        # Importer uses mesh basenames as USD identifiers; '-' is illegal.
        relative_path = Path(relative)
        local = ROOT / 'assets/meshes' / package / relative_path.with_name(relative_path.name.replace('-', '_'))
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, local)
        # Collada can refer to image textures; preserve their local relative paths.
        if original.suffix.lower() == '.dae':
            for image in ET.parse(original).findall('.//{*}image/{*}init_from'):
                texture = original.parent / image.text
                if not texture.is_file():
                    raise FileNotFoundError(f'DAE texture missing: {texture}')
                texture_local = local.parent / image.text
                texture_local.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(texture, texture_local)
                closure[str(texture_local.relative_to(ROOT))] = dict(path=str(texture_local.relative_to(ROOT)), sha256=sha(texture_local), bytes=texture_local.stat().st_size)
        closure[str(local.relative_to(ROOT))] = dict(path=str(local.relative_to(ROOT)), sha256=sha(local), bytes=local.stat().st_size,
                                                   source=str(original.relative_to(ROOT)))
        imported = local
        if original.suffix.lower() == '.dae':
            imported = local.with_suffix('.stl')
            count = collision_dae_to_stl(original, imported)
            closure[str(imported.relative_to(ROOT))] = dict(path=str(imported.relative_to(ROOT)), sha256=sha(imported), bytes=imported.stat().st_size,
                derived_from=str(local.relative_to(ROOT)), triangles=count, conversion='indexed triangles, unchanged raw ROS coordinates; no scene transforms')
        mesh.attrib['filename'] = '../' + str(imported.relative_to(ROOT))
    output = ROOT / 'urdf/tiago.urdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(document)
    ET.ElementTree(document).write(output, encoding='utf-8', xml_declaration=True)
    physics_document = copy.deepcopy(document)
    # ROS base_footprint is a massless reference frame. A free PhysX
    # articulation must root on the real massive chassis instead. Re-root
    # without changing any link's position relative to base_link.
    fixed_root = next(j for j in physics_document.findall('joint') if j.attrib['name']=='base_footprint_joint')
    offset = [float(x) for x in fixed_root.find('origin').attrib['xyz'].split()]
    if any(float(x) for x in fixed_root.find('origin').attrib.get('rpy','0 0 0').split()):
        raise ValueError('Unexpected rotating base_footprint transform')
    physics_document.remove(fixed_root)
    physics_document.remove(next(x for x in physics_document.findall('link') if x.attrib['name']=='base_footprint'))
    for joint in physics_document.findall('joint'):
        if joint.find('parent').attrib['link'] == 'base_footprint':
            joint.find('parent').attrib['link'] = 'base_link'
            origin = joint.find('origin')
            xyz = [float(x) for x in origin.attrib.get('xyz','0 0 0').split()]
            origin.attrib['xyz'] = ' '.join(str(x-y) for x,y in zip(xyz,offset))
    removed_empty = []
    for link in list(physics_document.findall('link')):
        name = link.attrib['name']
        if len(link) or any(j.find('parent').attrib['link']==name for j in physics_document.findall('joint')):
            continue
        parent_joint = next((j for j in physics_document.findall('joint') if j.find('child').attrib['link']==name),None)
        if parent_joint is not None and parent_joint.attrib['type']=='fixed':
            removed_empty.append(name)
            physics_document.remove(parent_joint)
            physics_document.remove(link)
    physics_output = ROOT/'urdf/tiago_physics.urdf'
    ET.ElementTree(physics_document).write(physics_output, encoding='utf-8', xml_declaration=True)
    srdf_entry = packages['tiago_moveit_config'] / 'config/srdf/tiago.srdf.xacro'
    srdf = xacro.process_file(str(srdf_entry), mappings=lock['configuration']).toxml()
    srdf_document = ET.fromstring(srdf)
    ET.indent(srdf_document)
    ET.ElementTree(srdf_document).write(ROOT/'urdf/tiago.srdf',encoding='utf-8',xml_declaration=True)
    licenses = []
    for source in lock['repositories']:
        original = sources/source['name']/'LICENSE'
        if original.is_file():
            local = ROOT/'LICENSES'/f'{source["name"]}_LICENSE'
            local.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(original,local)
            licenses.append(dict(path=str(local.relative_to(ROOT)),sha256=sha(local)))
    joints = []
    for joint in document.findall('joint'):
        if joint.attrib['type'] != 'fixed':
            limit = joint.find('limit')
            joints.append(dict(name=joint.attrib['name'], type=joint.attrib['type'],
                               parent=joint.find('parent').attrib['link'], child=joint.find('child').attrib['link'],
                               limits={} if limit is None else limit.attrib))
    manifest = dict(schema='robot_asset_manifest/v1', robot='PAL Robotics TIAGo steel, PMB2 differential base, single 7-DoF arm and lifting torso',
                    source_lock='SOURCE_LOCK.json', source_repositories=lock['repositories'], configuration=lock['configuration'],
                    entry='urdf/tiago_physics.urdf', sha256=sha(physics_output), original_resolved_urdf='urdf/tiago.urdf', original_resolved_sha256=sha(output),
                    physics_preparation=dict(root='base_link', removed_massless_ros_root='base_footprint', root_origin_m=offset, removed_empty_leaf_frames=removed_empty,
                                             geometry_preserved=True, actuator_tree_preserved=True),
                    links=len(document.findall('link')), joints=joints,
                    meshes=list(sorted(closure.values(), key=lambda item:item['path'])), downloaded_closure_complete=True,
                    removed_nonmechanical_tags=removed,
                    source_license_files=licenses,
                    license_note='Original PAL source repository LICENSE files are retained under assets/sources; inspect individual file headers.')
    (ROOT / 'ASSET_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(entry=str(ROOT/manifest['entry']), links=manifest['links'], actuated_joints=len(joints), meshes=len(closure)), indent=2))


if __name__ == '__main__':
    main()
