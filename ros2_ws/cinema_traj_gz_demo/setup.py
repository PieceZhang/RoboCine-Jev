from setuptools import find_packages, setup

package_name = "cinema_traj_gz_demo"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/demo_room.launch.py",
                                          "launch/frl_apartment_0.launch.py"]),
        (f"share/{package_name}/worlds", ["worlds/demo_room.world",
                                          "worlds/frl_apartment_0.world"]),
        (f"share/{package_name}/trajectories", ["trajectories/demo_room_retimed.json",
                                                "trajectories/frl_apartment_0_retimed.json",
                                                "trajectories/frl_apartment_0_subject.json"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Siyu Wang",
    maintainer_email="you@example.com",
    description="Gazebo Fortress (gz-sim6) demo for cinema_traj retimed trajectories.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "path_player = cinema_traj_gz_demo.path_player:main",
        ],
    },
)
