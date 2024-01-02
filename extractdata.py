#h5 file reader

import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi,sqrt
import scipy 
import scipy.constants

def gethdf5(filepath):
    return h5py.File(filepath,"r")


'''
def hdf5_to_dict(group):
    dict = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            dict[key] = hdf5_to_dict(item)
        else:
            dict[key] = item[()]
    return dict

def extract(path):
    
    if path!="":
        ex_data = h5py.File(path+'Fields0.h5', 'r')
        return hdf5_to_dict(ex_data)
    else:
        ex_data = h5py.File('Fields0.h5', 'r')
        return hdf5_to_dict(ex_data)
'''

'''
def hdf5_to_dict(group):
    """
    Recursively convert an HDF5 group and its contents into a dictionary,
    including attributes at each level.
    """
    result = {'attrs': dict(group.attrs)}  # Include attributes for the current group
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = hdf5_to_dict(item)
        else:
            result[key] = {'data': item[()], 'attrs': dict(item.attrs)}  # Include attributes for the dataset
    return result

def extract(file_path):
    """
    Convert an entire HDF5 file into a dictionary.
    """
    with h5py.File(file_path, 'r') as file:
        return hdf5_to_dict(file)

data_dict = extract("azim_4/Fields0.h5") 
'''


