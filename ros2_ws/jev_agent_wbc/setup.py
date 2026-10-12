from glob import glob
from pathlib import Path
from setuptools import setup

package_name = 'jev_agent_wbc'
setup(name=package_name, version='0.1.0', packages=[package_name],
      data_files=[('share/ament_index/resource_index/packages', ['resource/'+package_name]),
                  ('share/'+package_name, ['package.xml', 'VENDORED.json', 'README.md']),
                  ('share/'+package_name+'/scenarios', ['scenarios/catalog.json'])]+
                 [('share/'+package_name+'/'+folder, [p for p in glob(folder+'/*') if Path(p).is_file()]) for folder in
                  ['launch', 'config', 'urdf', 'meshes', 'scripts', 'jev_agent_wbc']]+
                 [('share/'+package_name+'/'+folder, glob(folder+'/*.json')) for folder in glob('scenarios/*') if Path(folder).is_dir()],
      install_requires=['setuptools', 'numpy', 'scipy', 'osqp'], zip_safe=False,
      maintainer='RoboCine', maintainer_email='robocine@example.com', license='MIT',
      description='Jev semantic task-space relaxation with dynamic cinematic whole-body control',
      entry_points={'console_scripts': ['cinematic_cpu=jev_agent_wbc.runner:main',
                                        'cinematic_ros=jev_agent_wbc.ros_node:main']})
