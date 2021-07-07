import matplotlib.pyplot as plt
import tqdm
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

from nuscenes.nuscenes import NuScenes

nusc_map = NuScenesMap(dataroot='./data/sets/nuscenes', map_name='singapore-onenorth')

fig, ax = nusc_map.render_layers(nusc_map.non_geometric_layers, figsize=1)

nusc_map.