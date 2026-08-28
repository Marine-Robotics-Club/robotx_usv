import os
from glob import glob
from setuptools import setup

package_name = "yolov26_ros"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),  # ✅ THIS
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="xavi",
    maintainer_email="xvicentnavar2024@fau.edu",
    description="YOLO26 (Ultralytics) ROS2 node: image topic -> detections + annotated image",
    license="MIT",
    entry_points={
        "console_scripts": [
            "detector = yolov26_ros.detector_ros:main",
            "webcam = yolov26_ros.webcam_node:main"
        ],
    },
)
