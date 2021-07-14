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

from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box

import numpy as np

from operator import itemgetter

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

    def get_closest_structures(self, layers, center_pose, max_objs=64, max_points=30, patch=[-1, -1],
                               global_coord=True, mode='within'):
        if patch[0] == -1:
            x_min = 0
            x_max = map_sizes[self.map_name][0]
        else:
            x_min = center_pose['translation'][0] - patch[0] / 2
            x_max = center_pose['translation'][0] + patch[0] / 2

        if patch[1] == -1:
            y_min = 0
            y_max = map_sizes[self.map_name][1]
        else:
            y_min = center_pose['translation'][1] - patch[1] / 2
            y_max = center_pose['translation'][1] + patch[1] / 2

        my_patch = [x_min, y_min, x_max, y_max]

        if layers is None:
            layers = self.non_geometric_layers

        output = list()
        for layer in layers:
            token_list = list(map(lambda x: x['token'], getattr(self, layer)))
            if layer in self.non_geometric_polygon_layers:
                layer_coords = list(map(lambda x: self.get_polygon_bounds(layer, x), token_list))

            elif layer in self.non_geometric_line_layers:
                layer_coords = list(map(lambda x: self.get_line_bounds(layer, x), token_list))

            layer_coords = list(filter(lambda x: len(x) > 0, layer_coords))
            layer_coords = list(filter(lambda x: self.intersect(x, my_patch), layer_coords))
            layer_coords = list(map(lambda x: self.nodes_abstraction(x, max_points=max_points), layer_coords))

            output += list(map(lambda x: {"layer": layer,
                                          "nodes": x,
                                          "min_dist": self.distance_from_ego(x, center_pose['translation'][:2])
                                          },
                               layer_coords
                               )
                           )
        output = sorted(output, key=itemgetter('min_dist'), reverse=False)
        output = output[:max_objs]
        if not global_coord:
            output = list(map(lambda x: {"layer": x['layer'],
                                          "nodes": self.transform_coord(x['nodes'], center_pose),
                                          "min_dist": x['min_dist']
                                          },
                               output
                               ))


        return output

    def intersect(self, node_coords, box_coords):
        x_min, y_min, x_max, y_max = box_coords
        cond_x = np.logical_and(node_coords[:, 0] < x_max, node_coords[:, 0] > x_min)
        cond_y = np.logical_and(node_coords[:, 1] < y_max, node_coords[:, 1] > y_min)
        cond = np.logical_and(cond_x, cond_y)
        return np.any(cond)

    def distance_from_ego(self, nodes, ego):
        diff_nodes = nodes - ego
        dist_array = np.sqrt((diff_nodes * diff_nodes).sum(axis=1))
        return dist_array[dist_array > 0].min()

    def nodes_abstraction(self, nodes, max_points=30):
        length = nodes.shape[0]
        output = np.zeros((30, 2))
        if length > 30:
            length = 30
            nodes = nodes[:30, :]

        output[:length, :] = nodes
        return output

    def transform_coord(self, nodes, coord):
        slicer = (nodes[:, 0] != 0) | (nodes[:, 1] != 0)
        temp = nodes[slicer, :]

        temp = temp.transpose(1, 0)
        temp = np.concatenate([temp, np.zeros([1, temp.shape[1]])], axis=0)

        car_from_global = transform_matrix(coord['translation'], Quaternion(coord['rotation']), inverse=True)

        temp = car_from_global.dot(np.vstack((temp, np.ones(temp.shape[1]))))[:3, :]
        temp = temp[:2, :]
        temp = temp.transpose(1, 0)

        nodes[slicer, :] = temp
        return nodes

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
    map_api = CustomNuScenesMap('../data/sets/nuscenes', 'boston-seaport')
    pose = dict([])

    pose['translation'] = [600.1202137947669, 1647.490776275174, 0.0]
    pose['rotation'] = [-0.968669701688471, -0.004043399262151301, -0.007666594265959211, 0.24820129589817977]
    structures = map_api.get_closest_structures(map_api.non_geometric_layers, pose, global_coord=False, max_objs=2048,
                                                max_points=1024, patch=[200, 200], mode='within')

    print(structures[0])
