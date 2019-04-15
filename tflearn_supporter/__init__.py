# | Created by Ar4ikov
# | Время: 15.04.2019 - 23:38

from tflearn_supporter.tflearn_supporter import TfLearnSupporter
from os import path

_version_file_path = path.join(path.dirname(__file__), "version")

with open(_version_file_path) as f:
    __version__ = f.readline().strip()
