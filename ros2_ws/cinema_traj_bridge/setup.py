from setuptools import find_packages, setup

package_name = "cinema_traj_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Siyu Wang",
    maintainer_email="you@example.com",
    description="Latched nav_msgs/Path server for cinema_traj exported camera trajectories.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "traj_server = cinema_traj_bridge.traj_server:main",
        ],
    },
)
