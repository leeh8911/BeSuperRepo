#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from glob import glob
import os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from pathlib import Path
import plotly.express as px
from multiprocessing import Pool
import multiprocessing as multi

def ground_truth_file_open(path):
    _columns = ['latDeg', 'lngDeg', 'heightAboveWgs84EllipsoidM']
    _df = pd.read_csv(path)    
    _df[['t_'+col for col in _columns]] = _df[_columns]
    _df = _df.drop(columns=_columns)
    return _df

def derived_file_open(path):
    _df = pd.read_csv(path)
    return _df
    
def gnsslog_file_open(path):
    gnss_section_names = {'Raw','UncalAccel', 'UncalGyro', 'UncalMag', 'Fix', 'Status', 'OrientationDeg'}
    _dict = gnss_log_to_dataframes(path)
    return _dict

def gnss_log_to_dataframes(path):
    gnss_section_names = {'Raw','UncalAccel', 'UncalGyro', 'UncalMag', 'Fix', 'Status', 'OrientationDeg'}
    with open(path) as f_open:
        datalines = f_open.readlines()

    datas = {k: [] for k in gnss_section_names}
    gnss_map = {k: [] for k in gnss_section_names}
    for dataline in datalines:
        is_header = dataline.startswith('#')
        dataline = dataline.strip('#').strip().split(',')
        # skip over notes, version numbers, etc
        if is_header and dataline[0] in gnss_section_names:
            try:
                gnss_map[dataline[0]] = dataline[1:]
            except:
                pass
        elif not is_header:
            try:
                datas[dataline[0]].append(dataline[1:])
            except:
                pass
    results = dict()
    for k, v in datas.items():
        results[k] = pd.DataFrame(v, columns=gnss_map[k])
    # pandas doesn't properly infer types from these lists by default
    for k, df in results.items():
        for col in df.columns:
            if col == 'CodeType':
                continue
            try:
                results[k][col] = pd.to_numeric(results[k][col])
            except:
                pass
    return results

