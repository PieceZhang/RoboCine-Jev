from setuptools import setup, find_packages

setup(
    name="vlm_intention",
    version="1.1.0",
    description="VLMIntention: obstacle-direction training-data pipeline + "
                "cinema_traj (3D map + motion prompt + subject pose → trajectory)",
    packages=find_packages(include=["vlm_intention", "vlm_intention.*",
                                    "cinema_traj", "cinema_traj.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "Pillow>=10.0",
        "PyYAML>=6.0",
        "requests>=2.31",
    ],
    extras_require={
        "depth": ["torch>=2.1", "transformers>=4.40"],
        "traj": ["torch>=2.1"],
        "viewer": ["opencv-python>=4.8"],
        "download": ["fiftyone>=0.24", "torchvision>=0.17"],
    },
)
