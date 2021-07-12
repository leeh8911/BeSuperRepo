# nuscenes-devkit
from nuscenes.nuscenes import NuScenes

# nuscenes-devkit-utils
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix

from pyquaternion import Quaternion

from functools import reduce
import os
import copy

import numpy as np


class CustomLidarApi():
    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
        self.token = None
        self.data = None

    def get_lidar_from_keyframe(self, token, max_points = 35000, car_coord = False):
        pc = np.zeros([4, 35000])

        if self.token == token:
            pass
        else:
            sample = self.nusc.get('sample', token)
            self.data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        pcl_path = os.path.join(self.nusc.dataroot, self.data['filename'])

        pc = LidarPointCloud.from_file(pcl_path)

        if car_coord:
            cs = self.nusc.get('calibrated_sensor', self.data['calibrated_sensor_token'])
            tf = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=True)
            pc.transform(tf)

        pc = self.abstract_point_cloud(pc, max_points)
        return pc

    def get_egopose_from_keyframe(self, token):

        if self.token == token:
            pass
        else:
            sample = self.nusc.get('sample', token)
            self.data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        return self.nusc.get("ego_pose", self.data['ego_pose_token'])

    def abstract_point_cloud(self, pc, max_points):
        point_shape = pc.points.shape
        if point_shape[1] > max_points:
            index = np.random.choice(point_shape[1], max_points, replace = False)
            pc.points = pc.points[:, index]
        else:
            pc.points = np.concatenate([pc.points, np.zeros((4, max_points - point_shape[1]))], axis = 1)
        return pc


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.express as px
    import tqdm

    nusc = NuScenes(version='v1.0-mini', dataroot='../data/sets/nuscenes', verbose=False)
    ldr_api = CustomLidarApi(nusc)

    my_scene = nusc.scene[0]
    first_sample_token = my_scene['first_sample_token']
    last_sample_token = my_scene['last_sample_token']

    token = first_sample_token

    my_sample = nusc.get('sample', token)
    first_time_1e6 = my_sample['timestamp']

    count = 0
    pc_list = []
    while (token != last_sample_token):
        if count > 5:
            break
        current_time_1e6 = my_sample['timestamp']
        time = (current_time_1e6 - first_time_1e6) * 1e-6

        pc = ldr_api.get_lidar_from_keyframe(token)
        temp = pd.DataFrame(pc.points.T, columns=['x', 'y', 'z', 'intensity'])
        temp['time'] = time
        pc_list.append(temp)

        token = my_sample['next']
        count += 1

    df = pd.concat(pc_list, axis=0)
    df['dummy'] = 1

    px.scatter(df, x='x', y='y', animation_frame='time', size='dummy', size_max=5, range_x=[-150, 150],
               range_y=[-150, 150])
