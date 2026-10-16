from glob import glob
from pathlib import Path
from setuptools import setup

package_name = 'jev_agent_wbc'
setup(name=package_name, version='0.2.0', packages=[package_name],
      data_files=[('share/ament_index/resource_index/packages', ['resource/'+package_name]),
                  ('share/'+package_name, ['package.xml', 'VENDORED.json', 'README.md', 'PAPER_DESIGN.md', 'VALIDATION.md', 'TPR_WBC.md', 'REPRESENTATION_DESIGN.txt', 'REPRESENTATION_VALIDATION.txt', 'REPRESENTATION_OPTIMIZATION.txt']),
                  ('share/'+package_name+'/scenarios', ['scenarios/catalog.json'])]+
                 [('share/'+package_name+'/'+folder, [p for p in glob(folder+'/*') if Path(p).is_file()]) for folder in
                  ['launch', 'config', 'urdf', 'meshes', 'scripts', 'jev_agent_wbc', 'schemas']]+
                 [('share/'+package_name+'/'+folder, glob(folder+'/*.json')) for folder in glob('scenarios/*') if Path(folder).is_dir()],
      install_requires=['setuptools', 'numpy', 'scipy', 'osqp'], zip_safe=False,
      maintainer='RoboCine', maintainer_email='robocine@example.com', license='MIT',
      description='Jev semantic task-space relaxation with dynamic cinematic whole-body control',
      entry_points={'console_scripts': ['cinematic_cpu=jev_agent_wbc.runner:main',
                                        'cinematic_camera=jev_agent_wbc.floating_camera:main',
                                        'cinematic_sam=jev_agent_wbc.sam_observer:main',
                                        'cinematic_ros=jev_agent_wbc.ros_node:main']})
