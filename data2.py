import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from pymatreader import read_mat
import os
from scipy import signal
from sklearn.model_selection import GroupShuffleSplit
import numpy_indexed as npi


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
    #Removes all the files that Lasse havent checked yet, so only using files with sub.
    filelist = [ x for x in filelist if "sub" in x ]
    return filelist


def unique_groups(main_path, filelist):
    # Initialize a list to store the unique subjects (for train test split)
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


def downsample(array, npts):
  # Downsample function
  downsampled = signal.resample(array, npts)
  return downsampled

def get_one_data(path, groupnr, groups):
  # Outputs one nd array in format (180, repetitions for one subject)
  data = read_mat(path)
  if "sub" in path:
      key = list(data.keys())[3]
  else:
      key = list(data.keys())[0]

  """
  ## Trying to make groups here. 
  if "sub" in path:
            subject = file[43:49]
        else:
            subject = file[39:45]
  if subject in filelist[k-1]:????
            i = i - 1

        if "sub" in file: 
            key = list(data.keys())[3]
        else:
            key = list(data.keys())[0]

        # make a list with group number of reps for each subject
        reps = data[key]["frames"]
        groups.extend([i]*reps)
        i += 1
  """

  X_raw = data[key]['values'][:,0]
  y = data[key]['frameinfo']['state']

  X_raw, y = delete_group456(X_raw,y)
  reps = len(y)
  groups.extend([groupnr]*reps)

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
    X_sliced.append(np.transpose(downsampled_X_raw)[i][4025:4110])
  X_sliced = np.transpose(np.array(X_sliced))

  X = X_sliced
  return X_raw,y, X, groups

def delete_group456(X,y):
  """Deleting frames with tag of 3,4,5 or 6"""
  indices_to_remove = [i for i in range(len(y)) if y[i] in [3, 4, 5, 6]]
  X = np.delete(np.transpose(X), indices_to_remove, axis=0)
  y = [y[i] for i in range(len(y)) if i not in indices_to_remove]
  X = np.transpose(X)
  return X, y

def get_all_data(filelist):
    '''Outputs X in shape (190, 2019 repitions for all subjects)''' 
    groups = []
    groupnr = 0
    X_raw, y, X_first, groups = get_one_data(filelist[0], groupnr, groups)
    filelist.pop(0)

    groupnr += 1
    for k, path in enumerate(filelist):
        
        #This part is making the belonging group number to the label.
        data = read_mat(path)
        if "sub" in path:
            subject = path[43:49]
            key = list(data.keys())[3]
        else:
            subject = path[39:45]
            key = list(data.keys())[0]

        if subject in filelist[k-1]:
            groupnr = groupnr - 1

          
        X_raw, y_loop, X, groups = get_one_data(path, groupnr, groups) #slice each subject
        groupnr += 1
        if len(X) != 0:
            X_first = np.concatenate((X, X_first),axis=1)
            y.extend(y_loop)
    X = X_first

    return X, y, groups
    
def group_split(X,y,groups):

  split = npi.group_by(groups).split(np.transpose(X))
  set(groups).tolist()
  C0 = np.transpose(split[0])
  C1 = np.transpose(split[1])
  AP_split, PA_split = C0.mean(1), C1.mean(1)
  return 1


def frame_split(X,y):
  C = npi.group_by(y).split(np.transpose(X))
  C0 = np.transpose(C[0])
  C1 = np.transpose(C[1])
  AP_split, PA_split = C0.mean(1), C1.mean(1)
  return AP_split, PA_split

def train_test_split(X,y,groups):
  #group_split = GroupShuffleSplit(X,y,groups)
  #return X_train, X_test, y_train, y_test
  return 1

def full_preprocess():
  main_path ="/mnt/projects/USS_MEP/COIL_ORIENTATION"
  filelist = data2.get_all_paths(main_path)
  X, y = data2.get_all_data(filelist)
  groups = data2.unique_groups(main_path, filelist)

  return X, y
#Herunder tester vi
import matplotlib.pyplot as plt
import numpy as np
import data2
import models


main_path ="/mnt/projects/USS_MEP/COIL_ORIENTATION"
filelist = data2.get_all_paths(main_path)
X, y, groups = data2.get_all_data(filelist)
#groups = data2.unique_groups(main_path, filelist)
#group_split(X,y,groups)
AP_split, PA_split = frame_split(X,y)
plt.plot(AP_split)
plt.plot(PA_split)
plt.show()

#logregscore, X_train, X_test, y_train, y_test, predictions = models.logregr(np.transpose(X),y)
#models.confmat(y_test, predictions, logregscore)



#filelist = get_all_paths(main_path)
#X,y  = get_all_data(filelist)
#print("bunitooo")
#print(X.shape)
#plt.plot(X)
#plt.show()
#print(X)
