from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
import os




if __name__ == "__main__":
    nusc = NuScenes(version='v1.0-mini', dataroot='./data/sets/nuscenes', verbose=True)
    scene_list = nusc.list_scenes()

    my_scene = nusc.scene[0]
    first_sample_token = my_scene['first_sample_token']

    my_sample = nusc.get('sample', first_sample_token)

    lidar_top_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
    pcl_path = os.path.join(nusc.dataroot, lidar_top_data['filename'])
    pc = LidarPointCloud.from_file(pcl_path)
    print(pc.points)