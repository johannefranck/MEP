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

def get_one_data(path, groupnr, groups):
  # Outputs one nd array in format (180, repetitions for one subject)
  data = read_mat(path)
  if "sub" in path:
      key = list(data.keys())[3]
  else:
      key = list(data.keys())[0]

  X_raw = data[key]['values'][:,0]
  y = data[key]['frameinfo']['state']
  X_raw, y = delete_frames(X_raw,y)
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

def delete_frames(X,y):
  """Deleting frames with tag of 3,4,5 or 6, PA: 1, AP: 2"""
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
    

def plotgroups(X, groups):
  colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # get a list of colors from the default color cycle in matplotlib
  color_dict = {i: colors[i%len(colors)] for i in range(52)} # create a dictionary with keys from 0 to 51, and values as different colors from the matplotlib library

  # Get the time array with a sampling rate of 2000 Hz
  STAA = 7.5 # sliced_time_after_artifact, here it is 15 timepoints, which is the same a 7.5 ms
  time = np.arange(np.transpose(X).shape[1]) / 2000 * 1000 + STAA

  # Create the figure and axes objects
  fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

  # Plot the data for each group
  lines = []
  for group in set(groups):
      idx = np.where(np.array(groups) == group)[0]
      for i in range(len(idx)):
          line, = axs.plot(time, np.transpose(X)[idx[i], :], c=color_dict[group], alpha=0.5)
          lines.append(line)

  # Set the axis labels and title
  axs.set_xlabel('Time (ms)')
  axs.set_ylabel('MEP signal')
  axs.set_title('MEP signals by group')

  # Add the legend
  legend_labels = list(set(groups))
  axs.legend(lines, legend_labels, loc='best', prop={'size': 'xx-small'}, title='Group', title_fontsize='small', framealpha=0.5, facecolor='white', edgecolor='black', labelcolor=colors)

  # Show the plot
  plt.show()
#filelist = get_all_paths(main_path)'xx-small',
#X,y  = get_all_data(filelist)
#print("bunitooo")
#print(X.shape)
#plt.plot(X)
#plt.show()
#print(X)
