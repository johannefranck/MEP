print("hej johanne er sej")
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from pymatreader import read_mat
import os
from scipy import signal
import data

# Define the path
main_path ="/mnt/projects/USS_MEP/COIL_ORIENTATION"
filelist = data.get_all_paths(main_path)

def unique_groups(main_path, filelist):
    # Initialize a list to store the unique subjects (to use as groups)
    groups = []
    i = 0
    for k, file in enumerate(filelist):
        subject = file[43:49]

        if subject in filelist[k-1]:
            i = i - 1

        data = read_mat(file)
        key = list(data.keys())[3]

        reps = data[key]["frames"]
        groups.extend([i]*reps)
        i += 1

    return groups

groups = unique_groups(main_path, filelist)
print(groups)
print(len(groups))