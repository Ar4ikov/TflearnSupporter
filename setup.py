# | Created by Ar4ikov
# | Время: 16.04.2019 - 01:52

from setuptools import setup

import os

_version_file_path = os.path.join(os.path.dirname(__file__), "tflearn_supporter", "version")
_readme_file_path = os.path.join(os.path.dirname(__file__), "README.md")

with open(_version_file_path) as f:
    __version__ = f.readline().strip()

with open(_readme_file_path, "rb") as f:
    readme = f.read().decode("utf-8")

setup(
    name="tflearn_supporter",
    version=__version__,
    install_requires=[],
    long_description=readme,
    packages=["tflearn_supporter"],
    package_data={"tflearn_supporter": ["version"]},
    url="https://github.com/Ar4ikov/TfLearnSupporter",
    license="MIT License",
    author="Nikita Archikov",
    author_email="bizy18588@gmail.com",
    description="An easiest way to save your Tensorflow model with your trained data in one directory to fast "
                "continue use it in future",
    keywords="opensource machine learning deep tensorflow tensor tflearn supporter support"
)
