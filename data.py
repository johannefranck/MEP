import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from pymatreader import read_mat
import os
from scipy import signal

#path to all the files
main_path ="/mnt/projects/USS_MEP/COIL_ORIENTATION"
path_x01666 = "/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X01666_ses-1_task-coilorientation_emg.mat"


def get_all_paths(main_path):
    # Outputs Filelist containing all files
    #Store all the file names in this list
    filelist = []

    #Making a list of all the files paths to all the data files. 
    for root, dirs, files in os.walk(main_path):
        for file in files:
            #append the file name to the list
            filelist.append(os.path.join(root,file))
    #Removes xlsx files
    filelist = [ x for x in filelist if "xlsx" not in x ] #50
    #filelist = [ x for x in filelist if "sub" not in x ] #19
    
    return filelist


def unique_groups(main_path, filelist):
    # Initialize a list to store the unique subjects (for train test split)
    groups = []
    i = 0
    for k, file in enumerate(filelist):
        if "sub" in file:
            subject = file[43:49]
        else:
            subject = file[39:45]
        #print(subject)
        # count subject as one group in case of multiple runs
        if subject in filelist[k-1]:
            i = i - 1

        #we need the key
        data = read_mat(file)
        if "sub" in file: 
            key = list(data.keys())[3]
        else:
            key = list(data.keys())[0]

        # make a list with group number number of reps for each subject
        reps = data[key]["frames"]
        groups.extend([i]*reps)
        i += 1
    return groups

#filelist = get_all_paths(main_path)
#groups = unique_groups(main_path, filelist)


def downsample(array, npts):
  # Downsample function
  downsampled = signal.resample(array, npts)
  return downsampled

def get_one_data(path):
  # Outputs one nd array in format (180, repetitions for one subject)
  data = read_mat(path)
  if "sub" in path:
      key = list(data.keys())[3]
  else:
      key = list(data.keys())[0]

  X_raw = data[key]['values'][:,0]
  y = data[key]['frameinfo']['state']
  X_raw, y = delete_group456(X_raw,y)

  # downsample
  if len(X_raw)==20000:
    downsampled_X_raw = []
    for i in range(len(X_raw[0])):
      downsampled_X_raw.append(downsample(np.transpose(X_raw)[i], 8000).tolist())
    downsampled_X_raw = np.transpose(downsampled_X_raw)
  else:
    downsampled_X_raw = X_raw

  # slice MEP 
  X_sliced = []
  for i in range(len(np.transpose(downsampled_X_raw))):
    X_sliced.append(np.transpose(downsampled_X_raw)[i][4010:4200])
  X_sliced = np.transpose(np.array(X_sliced))

  X = X_sliced
  return X_raw,y, X

def delete_group456(X,y):
  """Deleting frames with tag of 4,5 or 6"""
  indices_to_remove = [i for i in range(len(y)) if y[i] in [3, 4, 5, 6]]
  X = np.delete(np.transpose(X), indices_to_remove, axis=0)
  y = [y[i] for i in range(len(y)) if i not in indices_to_remove]
  X = np.transpose(X)
  return X, y

def get_all_data(filelist):
    '''Outputs X in shape (190, 2019 repitions for all subjects)''' 
    X_raw, y, X_first = get_one_data(filelist[0])
    filelist.pop(0)

    for path in filelist:

        X_raw, y_loop, X = get_one_data(path) #slice each subject
        if len(X) != 0:
            X_first = np.concatenate((X, X_first),axis=1)
            y.extend(y_loop)
    X = X_first

    return X, y
    


#filelist = get_all_paths(main_path)
#X,y  = get_all_data(filelist)
#print("bunitooo")
#print(X.shape)
#plt.plot(X)
#plt.show()
#print(X)
