import os
from glob import glob

from setuptools import find_packages, setup


package_name = "jvs_mock"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md", "STAGE1_WBC.md", "CAMERA_SHOTS.md", "STAGE2_SHOT_WBC.md", "STAGE3_KC_WBC.md", "LIDAR_MAP_VALIDATION.md", "DUAL_LIDAR_MESH.md", "LIDAR_WBC.md", "DYNAMIC_WBC.md"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml") + glob("config/*.rviz") + glob("config/*.json"),
        ),
        (os.path.join("share", package_name, "meshes"), glob("meshes/*")),
        (os.path.join("share", package_name, "scripts"), glob("scripts/*.py")),
        (os.path.join("share", package_name, "trajectories"), glob("trajectories/*.json")),
        (os.path.join("share", package_name, "scenes"), glob("scenes/*.json")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*.xacro")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.world") + glob("worlds/*.usda")),
    ],
    install_requires=["setuptools", "numpy", "scipy", "osqp>=0.6.7,<2"],
    zip_safe=True,
    maintainer="Yuelin",
    maintainer_email="you@example.com",
    description="RoboCine differential-drive and 6-DOF camera-arm mock robot.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "base_velocity_commander = jvs_mock.base_velocity_commander:main",
            "controller_status = jvs_mock.controller_status:main",
            "isaac_command_adapter = jvs_mock.isaac_command_adapter:main",
            "isaac_reset_client = jvs_mock.isaac_reset_client:main",
            "joint_space_commander = jvs_mock.joint_space_commander:main",
            "jvs_follow_traj = jvs_mock.jvs_follow_traj:main",
            "whole_body_controller = jvs_mock.whole_body_controller:main",
            "dynamic_whole_body_controller = jvs_mock.dynamic_whole_body_controller:main",
        ],
    },
)
