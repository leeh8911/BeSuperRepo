
# nuscenes-devkit
from nuscenes.nuscenes import NuScenes

# nuscenes-devkit-utils
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix

from pyquaternion import Quaternion

from functools import reduce
import os

import numpy as np


class CustomLidarApi():
    def __init__(self, nusc:NuScenes):
        self.nusc = nusc

    def get_lidar_from_keyframe(self, token):
        pc = np.zeros([4, 35000])

        return pc