import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn import metrics
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC

from pymatreader import read_mat
import os
from scipy import signal
import numpy_indexed as npi


#path to all the files
main_path ="/mnt/projects/USS_MEP/COIL_ORIENTATION"
path_x01666 = "/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X01666_ses-1_task-coilorientation_emg.mat"


def get_all_paths(main_path):
    # Outputs Filelist containing all paths to files
    # Store all the file names in this list
    filelist = []

    # Making a list of all the files paths to all the data files. 
    for root, dirs, files in os.walk(main_path):
        for file in files:
            # Append the file name to the list
            filelist.append(os.path.join(root,file))
    
    # Removes xlsx files
    filelist = [ x for x in filelist if "xlsx" not in x ] 
    filelist = [ x for x in filelist if "02035" not in x ] #bad signal
    filelist = [ x for x in filelist if "96343" not in x ] #bad signal

    filelist = np.sort(filelist).tolist()
    return filelist

def downsample(array, npts):
    # Downsample function, returns array as downsampled
    downsampled = signal.resample(array, npts)
    return downsampled

def get_one_data(path, groupnr, groups):
    # Outputs one nd array in format (85 points, repetitions for one subject, list of all groups)
    # Uses the downsampel 
    # Uses the delete frames
    # Does slicing to get the time with MEP
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
    
    # Downsample
    if len(X_raw)==20000:
      downsampled_X_raw = []
      for i in range(len(X_raw[0])):
        downsampled_X_raw.append(downsample(np.transpose(X_raw)[i], 8000).tolist())
      downsampled_X_raw = np.transpose(downsampled_X_raw)
    else:
      downsampled_X_raw = X_raw

    # Slice signal to specific range with MEP
    X_sliced = []
    for i in range(len(np.transpose(downsampled_X_raw))):
        # for some paths the MEP signal is at 5000/20k, so the cut should be different
        if "sub" not in path:
            X_sliced.append(np.transpose(downsampled_X_raw)[i][2025:2110]) #cutter måskeke for meget af
        else:
            X_sliced.append(np.transpose(downsampled_X_raw)[i][4025:4110])
            
    X_sliced = np.transpose(np.array(X_sliced))
    X = X_sliced

    return X_raw,y, X, groups

def delete_frames(X,y):
    # Deleting frames with tag of 3,4,5 or 6. PA: 1, AP: 2
    indices_to_remove = [i for i in range(len(y)) if y[i] in [3, 4, 5, 6]]
    X = np.delete(np.transpose(X), indices_to_remove, axis=0)
    y = [y[i] for i in range(len(y)) if i not in indices_to_remove]
    X = np.transpose(X)
    return X, y

def get_all_data(filelist):
    # Outputs X in shape (85 points signal, 2019 repitions for all subjects)??
    groups = []
    groupnr = 0
    filelist_idx0 = filelist[0]
    X_raw, y, X_first, groups = get_one_data(filelist_idx0, groupnr, groups)
    filelist.pop(0)
    
    list_subjects = [] # names
    groupnr += 1
    for k, path in enumerate(filelist):
        # This part is making the belonging group number to the label.
        # I.e. checking whether this subject is the same as the prevouis
        data = read_mat(path)
        if "sub" in path:
            subject = path[43:49]
            key = list(data.keys())[3]
        else:
            subject = path[39:45]
            key = list(data.keys())[0]

        if subject in filelist[k-1]:
            groupnr = groupnr - 1

        # Slice each subject
        X_raw, y_loop, X, groups = get_one_data(path, groupnr, groups) 
        groupnr += 1
        if len(X) != 0:
            X_first = np.concatenate((X, X_first),axis=1)
            y.extend(y_loop)
        
        if subject not in list_subjects:
            list_subjects.append(subject)
    list_subjects.insert(0,filelist_idx0[39:45])
    filelist.insert(0,filelist_idx0)
    X = X_first

    filelist = np.sort(filelist).tolist()

    return X, y, groups, list_subjects
    


def plot_groups(X, groups, specifics):
    # Plotting the different subject MEP signals in the same plot, with time on the x axis

    # Color pre-definitions
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # get a list of colors from the default color cycle in matplotlib
    color_dict = {i: colors[i%len(colors)] for i in range(52)} # create a dictionary with keys from 0 to 51, and values as different colors from the matplotlib library

    # Get the time array with a sampling rate of 2000 Hz
    STAA = 7.5 # sliced_time_after_artifact, here it is 15 timepoints, which is the same a 7.5 ms
    time = np.arange(np.transpose(X).shape[1]) / 2000 * 1000 + STAA

    # Create the figure and axes objects
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    # Plot the data for each group
    lines = []
    if specifics != ["all"]:
        idx = 0
        for group in set(specifics):
            traj_ids = np.where(np.array(groups) == group)[0]
            for i in range(len(traj_ids)):
                line, = axs.plot(time, np.transpose(X)[traj_ids[i], :], c=color_dict[group+idx], alpha=0.5)
                lines.append(line)
            idx += 1
            colors
    else:
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
    if specifics != ["all"]:
        groups = specifics
    legend_labels = list(set(groups))
    axs.legend(lines, legend_labels, loc='best', prop={'size': 'xx-small'}, title='Group', title_fontsize='small', framealpha=0.5, facecolor='white', edgecolor='black', labelcolor=colors)

    # Show the plot
    plt.show()

def frame_split(X,y):
    # Splitting the frames (AP and PA), and plotting their mean compared to each other
    C = npi.group_by(y).split(np.transpose(X))
    C0 = np.transpose(C[0])
    C1 = np.transpose(C[1])
    AP_split, PA_split = C0.mean(1), C1.mean(1)
    return AP_split, PA_split

def train_test_split(X,y,groups):
    from sklearn.model_selection import LeaveOneGroupOut
    X = np.array(np.transpose(X))
    y = np.array(y)
    groups = np.array(groups)
    group_kfold = LeaveOneGroupOut()
    #group_kfold.get_n_splits(X, y, groups)
    #print(group_kfold)
    GroupKFold(n_splits=2)
    scores = []
    for train_index, test_index in group_kfold.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        """
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        accuracy = lr.score(X_test, y_test)
        scores.append(accuracy)
        print(accuracy, scores)
        """

    """
    group_split = GroupShuffleSplit(n_splits=43, test_size=0.1, random_state=None)
    group_split.get_n_splits()
    for i, (train_index, test_index) in enumerate(group_split.split(np.transpose(X), y, groups)):
      print(f"Fold {i}:")
      print(f"  Train: index={train_index}, group={groups[train_index]}")
      print(f"  Test:  index={test_index}, group={groups[test_index]}")
    #group_split = GroupShuffleSplit(X,y,groups)
    """

    return X_train, X_test, y_train, y_test

def other_X(X):
    X_amplitude = np.max(X, axis=0) - np.min(X, axis=0)
    X_amplitude = X_amplitude.reshape(-1, 1)

    max_indices = np.argmax(X, axis=0)
    X_latency = max_indices.reshape(-1, 1)
    # Concatenate X_amplitude and max_indices_column horizontally
    X_ampl_late = np.hstack((X_amplitude, X_latency))


    #normalized X
    # compute minimum and maximum values along columns, because we have signals on the columns so we dont need to transpose
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)

    # normalize X using min-max scaling for each column
    X_norm = (X - min_val) / (max_val - min_val)


    return X_amplitude, X_latency,X_ampl_late, X_norm