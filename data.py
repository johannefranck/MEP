import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from pymatreader import read_mat
import os
from scipy.interpolate import interp1d

#path to all the files
main_path ="/mnt/projects/USS_MEP/COIL_ORIENTATION"
path_x01666 = "/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X01666_ses-1_task-coilorientation_emg.mat"

# Outputs Filelist containing all files
def get_all_paths(main_path):
    #Store all the file names in this list
    filelist = []

    #Making a list of all the files paths to all the data files. 
    for root, dirs, files in os.walk(main_path):
        for file in files:
            #append the file name to the list
            filelist.append(os.path.join(root,file))
    #Removes all the files that Lasse havent checked yet, so only using files with sub.
    filelist = [ x for x in filelist if "sub" in x ]
    return filelist

from scipy import signal
# Downsample function
def downsample(array, npts):
    #interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    #downsampled = interpolated(np.linspace(0, len(array), npts))
    downsampled = signal.resample(array, npts)
    return downsampled

# Outputs one nd array in format (180, repetitions)
def get_one_data(path):
  data = read_mat(path)
  key = list(data.keys())[3]

  X_raw = np.transpose(data[key]['values'][:,0])
  y = data[key]['frameinfo']['state']
  
  # downsample
  
  if len(X_raw[0])==20000:
    downsampled_X_raw = X_raw
    for i in range(len(X_raw)):
      downsampled_X_raw[i] = downsample(X_raw[i], 8000)
  else:
    downsampled_X_raw = X_raw
  #plt.plot(downsampled_X_raw)
  #plt.show()

  X_sliced = []
  #OBS Vurder hvornår vi skal slice fra og til. 
  x_start, x_end = int(len(downsampled_X_raw[0,:])/2+20) , int(len(downsampled_X_raw[0,:])/2+200)
  for i in range(len(downsampled_X_raw)):
    X_sliced.append(downsampled_X_raw[i,:][x_start:x_end])
  X_sliced = np.transpose(np.array(X_sliced))

  X = X_sliced
  return X_raw,y, X

def get_all_data(filelist):
    X_raw, y, X_first = get_one_data(filelist[0])
    filelist.pop(0)

    for path in filelist:
        #data = read_mat(path)
        #key = list(data.keys())[3]

        X_raw, y, X = get_one_data(path) #slice each subject
        X_first = np.concatenate((X, X_first),axis=1)

    X = X_first
    return X
    


filelist = get_all_paths(main_path)
X = get_all_data(filelist)
print(X.shape)
plt.plot(X)
plt.show()
#print(X)
