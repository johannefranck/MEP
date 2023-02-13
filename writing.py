import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from pymatreader import read_mat
path1 = '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X24208_ses-1_task-coilorientation_run001_emg.mat'
#path1 = '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X01666_ses-1_task-coilorientation_emg.mat'
#path1 = '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X02299_ses-1_task-coilorientation_emg.mat'
#path1 = '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X03004_ses-1_task-coilorientation_emg.mat'
#key = 'X02299_Coil_Orient000_1_wave_data'
data = read_mat(path1)
key = list(data.keys())[3]

import os

folderpath = r"/mnt/projects/USS_MEP/test"
filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]
all_files = []

for path in filepaths:
    with open(path, 'r') as f:
        file = f.readlines()
        all_files.append(file)

print(all_files)
