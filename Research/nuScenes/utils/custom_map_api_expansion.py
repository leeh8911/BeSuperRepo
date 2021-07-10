from typing import Dict, List

from _ctypes import LoadLibrary as _dlopen

_dlopen("D:\\Sangwons_Room\\00_SoftWares\\Anaconda\\envs\\torch\\Library\\bin\\geos.dll", 0)

# nuscenes-map expansion
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

# nuscenes utils
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion import Quaternion

import numpy as np

map_sizes = {'singapore-onenorth': [1585.6, 2025.0],
             'singapore-hollandvillage': [2808.3, 2922.9],
             'singapore-queenstown': [3228.6, 3687.1],
             'boston-seaport': [2979.5, 2118.1],
             }

locations = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']


class CustomNuScenesMap(NuScenesMap):
    def __init__(self, dataroot='./data/sets/nuscenes', map_name='singapore-onenorth'):
        super().__init__(dataroot=dataroot, map_name=map_name)
        self.explorer = NuScenesMapExplorer(self)

    def get_size(self):
        return map_sizes[self.map_name]

    def get_closest_layers(self, layers, center_pose, patch=[200, 200], mode='within'):
        my_patch = [center_pose[0] - patch[0] / 2, center_pose[1] - patch[1] / 2, center_pose[0] + patch[0] / 2, center_pose[1] + patch[1] / 2]
        records = self.get_records_in_patch(my_patch, layers, mode='within')
        output = dict()

        for layer in layers:
            if layer in self.non_geometric_polygon_layers:
                tokens = records[layer]
                layer_coords = []
                for token in tokens:
                    node_coords = self.get_polygon_bounds(layer, token)
                    layer_coords.append(node_coords)

            elif layer in self.non_geometric_line_layers:
                tokens = records[layer]
                layer_coords = []
                for token in tokens:
                    node_coords = self.get_line_bounds(layer, token)
                    layer_coords.append(node_coords)
            else:
                continue
            output[layer] = layer_coords

        return output

    def get_closest_structures(self, layers, center_pose, max_objs=64, max_points=30, patch=[200, 200],
                               global_coord = True, mode='within'):
        layer_dict = self.get_closest_layers(layers, center_pose['translation'], patch, mode)
        output_list = []
        for layer in layers:
            nodes_list = layer_dict[layer]
            for nodes in nodes_list:
                nodes = self.nodes_abstraction(nodes)

                if not global_coord:
                    nodes = self.transform_coord(nodes, center_pose)

                struct_dict = {'class': layer, 'nodes': nodes}
                output_list.append(struct_dict)

        return output_list

    def nodes_abstraction(self, nodes):
        output = np.zeros((2, 30))
        length = nodes.shape[0]
        if length > 30:
            nodes = nodes[:30, :]
        return nodes.transpose(1, 0)

    def transform_coord(self, nodes, coord):
        nodes = np.concatenate([nodes, np.zeros([1, nodes.shape[1]])], axis=0)

        car_from_global = transform_matrix(coord['translation'], Quaternion(coord['rotation']), inverse=True)

        nodes = car_from_global.dot(np.vstack((nodes, np.ones(nodes.shape[1]))))[:3, :]
        output = nodes[:2, :]

        return output

    def get_polygon_bounds(self, layer, token):

        record = self.get(layer, token)
        if 'exterior_node_tokens' in record.keys():
            nodes = [self.get('node', token) for token in record['exterior_node_tokens']]
            node_coords = np.array([(node['x'], node['y']) for node in nodes])
        else:
            node_coords = np.array([])

        return node_coords

    def get_line_bounds(self, layer, token):

        record = self.get(layer, token)
        nodes = [self.get('node', token) for token in record['node_tokens']]
        node_coords = np.array([(node['x'], node['y']) for node in nodes])

        return node_coords


if __name__ == "__main__":
    map_api = CustomNuScenesMap('../data/sets/nuscenes', 'singapore-onenorth')
    structures = map_api.get_closest_structures(['road_divider', 'road_block', 'walkway', 'traffic_light'], [600, 1000])
    layers = map_api.get_closest_layers(['road_divider', 'road_block', 'walkway', 'traffic_light'], [600, 1000])

    print(structures)
    print(layers)
