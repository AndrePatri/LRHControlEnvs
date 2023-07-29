"""Installation script for the 'omnicustomgym' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # RL
    "torch==1.12.0",
    "gym==0.26.2",
    "gymnasium==0.28.1", 
    "stable_baselines3[extra]==2.0.0a10", 
    "box2d-py",
    "tensorboard",
    "tensorboard-plugin-wit",
    "protobuf",
    "termcolor",
    "hydra-core>=1.1",
    "rospkg",
    ]

# Installation operation
setup(
    name="omnicustomgym",
    author="AndPatr",
    version="1.0.0",
    description="",
    keywords=["omnicustomgym", "stepping", "rl"],
    include_package_data=True,
    python_requires=">=3.7, <3.8",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.6, 3.7, 3.8"],
    zip_safe=False,
)

# EOF
