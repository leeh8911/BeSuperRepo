import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from nuscenes.nuscenes import NuScenes
from custom_lidar_api import CustomLidarApi
from custom_map_api_expansion import CustomNuScenesMap

plt.style.use('seaborn-bright')


class NusceneDataHelper():
    def __init__(self, version,
                 dataroot='./data/sets/nuscenes',
                 locations=['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport'],
                 target_layer_names=['road_block', 'walkway', 'road_divider', 'traffic_light'],
                 max_objs=64,
                 max_points=30,
                 max_point_cloud=1200,
                 patch_size=[-1, -1]):
        self.class_names = ["None"] + target_layer_names
        self.dataroot = dataroot
        self.locations = locations
        self.target_layer_names = target_layer_names
        self.max_objs = max_objs
        self.max_points = max_points
        self.max_point_clouds = max_point_cloud
        self.patch_size = patch_size

        self.class_dict = dict()
        for i, name in enumerate(self.target_layer_names):
            self.class_dict[name] = i + 1
        self.class_array = np.eye(len(self.class_names))

        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.ldr_api = CustomLidarApi(self.nusc)
        self.map_api = dict()
        for loc in locations:
            self.map_api[loc] = CustomNuScenesMap(dataroot=dataroot, map_name=loc,
                                                  target_layer_names=target_layer_names, max_objs=max_objs,
                                                  max_points=max_points)

    def get_frame_from_token(self, token):
        sample = self.nusc.get('sample', token)
        scene = self.nusc.get('scene', sample['scene_token'])
        log_meta = self.nusc.get('log', scene['log_token'])

        location = log_meta['location']
        sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        pc = self.ldr_api.get_lidar_from_keyframe(token, max_points=self.max_point_clouds, car_coord=True)
        ego = self.ldr_api.get_egopose_from_keyframe(token)
        structures = self.map_api[log_meta['location']].get_closest_structures(ego,
                                                                               patch=self.patch_size,
                                                                               global_coord=False,
                                                                               mode='intersect'
                                                                               )
        return pc, structures, ego

    def get_all_sample_tokens(self):
        sample_tokens = []
        for scene in self.nusc.scene:
            token = scene['first_sample_token']
            while token != scene['last_sample_token']:
                sample_tokens.append(token)
                sample = self.nusc.get('sample', token)
                token = sample['next']
        return sample_tokens

    def get_label(self, structures):
        classes = list(map(lambda x: np.array(self.class_array[self.class_dict[x["layer"]], :]).reshape(len(self.class_names),1),structures))
        classes = np.concatenate(classes, axis=1)

        objects = list(map(lambda x: np.array(x['nodes'].reshape(1, self.max_points, 2)), structures))
        objects = np.concatenate(objects, axis=0)

        if len(structures) < self.max_objs:
            diff = self.max_objs - len(structures)
            classes = np.concatenate([classes, np.zeros((5, diff))], axis = 1)
            objects = np.concatenate([objects, np.zeros((diff, 30, 2))], axis = 0)
            
        return classes, objects

    def draw_data(self,
                  token,
                  ax=None,
                  model=None,
                  view_2d=True,
                  save_name=""):
        pc, structures, ego = self.get_frame_from_token(token)

        pc_array = pc.points
        if model is None:
            pred_class = np.array([])
            pred_pose = np.array([])
        else:
            X = torch.Tensor(np.expand_dims(pc_array, axis=0)).to(next(model.parameters()).device)
            model.eval()
            pred_class, pred_pose = model(X)
            pred_class = pred_class.squeeze(0).cpu().detach().numpy()
            pred_pose = pred_pose.squeeze(0).cpu().detach().numpy()
        true_class, true_pose = self.get_label(structures)

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 16))

        ax.plot(pc_array[0, :], pc_array[1, :], "o", linestyle="", label='lidar point cloud')
        for i in range(pred_pose.shape[0]):
            cls_ = pred_class[:, i].argmax()
            index = (pred_pose[i, :, 0] != 0) & (pred_pose[i, :, 1] != 0)
            ax.plot(pred_pose[i, index, 0], pred_pose[i, index, 1], "-")
            ax.text(pred_pose[i, 0, 0], pred_pose[i, 0, 1], self.class_names[cls_], fontsize = 'x-large')

        for i in range(true_pose.shape[0]):
            cls_ = true_class[:, i].argmax()
            index = (true_pose[i, :, 0] != 0) & (true_pose[i, :, 1] != 0)
            ax.plot(true_pose[i, index, 0], true_pose[i, index, 1], ":")
            ax.text(true_pose[i, 0, 0], true_pose[i, 0, 1], self.class_names[cls_], fontsize = 'x-large')
        
        ax.set_xlim([-60, 60])
        ax.set_ylim([-60, 60])

        if save_name != "":
            save_path, _ = os.path.split(save_name)
            if os.path.isdir(save_path):
                pass
            else:
                os.mkdir(save_path)
            plt.savefig(save_name)
            plt.close(fig)


if __name__ == "__main__":
    nusc_helper = NusceneDataHelper(version='v1.0-trainval',
                                    dataroot='E:/datasets/nuscenes',
                                    target_layer_names=['road_block', 'walkway', 'road_divider', 'traffic_light'],
                                    max_objs=4,
                                    max_points=30,
                                    max_point_cloud=1200,
                                    patch_size=[-1, -1]
                                    )

    tokens = nusc_helper.get_all_sample_tokens()
    print(len(tokens))

    pc, structures, ego = nusc_helper.get_frame_from_token(tokens[0])
    print(pc.points.shape)

    classes, objects = nusc_helper.get_label(structures)
    print(classes.shape)
    print(objects.shape)

    nusc_helper.draw_data(token=tokens[0])
    plt.show()
