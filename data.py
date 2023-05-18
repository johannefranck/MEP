import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn import metrics
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
import plot_functions
from pymatreader import read_mat
import os
from scipy import signal
import numpy_indexed as npi

import torch
import torch.nn as nn



#path to all the files
#main_path ="/mnt/projects/USS_MEP/COIL_ORIENTATION"
#path_x01666 = "/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X01666_ses-1_task-coilorientation_emg.mat"


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
    filelist = [ x for x in filelist if "02035" not in x ] #noisy signal    
    filelist = [ x for x in filelist if "06188" not in x ] #no AP signal + flat top 
    filelist = [ x for x in filelist if "91583" not in x ] #noisy signal    
    #filelist = [ x for x in filelist if "71487" not in x ] #old noisy signal
    #filelist = [ x for x in filelist if "34646" not in x ] #old noisy signal
    #filelist = [ x for x in filelist if "40027" not in x ] #old noisy signal

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

        #print(f"Processing subject: {subject}, index: {k}, groupnr: {groupnr}")
        # Slice each subject
        X_raw, y_loop, X, groups = get_one_data(path, groupnr, groups) 
        groupnr += 1
        if len(X) != 0:
            X_first = np.concatenate((X_first, X),axis=1)
            y.extend(y_loop)
        
        if subject not in list_subjects:
            list_subjects.append(subject)
    list_subjects.insert(0,filelist_idx0[39:45])
    filelist.insert(0,filelist_idx0)
    X = X_first

    filelist = np.sort(filelist).tolist()

    return X, y, groups, list_subjects
    

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

    # Diffentiate X, Note that the first element of each row will be lost after differentiation since there is no previous element to calculate the difference with.    X_diff = np.diff(X, axis=1)
    X_diff = np.diff(X, axis=1)



    fs = 2000  # Sample rate (Hz)

    
    """for i in range(np.transpose(X).shape[0]):
        # Compute the power spectrum for signal i
        ps = np.abs(np.fft.rfft(np.transpose(X)[i])) ** 2
        
        # Compute the corresponding frequencies
        freqs = np.fft.rfftfreq(len(np.transpose(X)[i]), d=1/fs)
        
        # Plot the power spectrum for signal i
        plt.plot(freqs, ps)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title(f'Power spectrum for signal {i+1}')
        plt.show()"""
    X_fft = np.fft.fft(X, axis=0)# virker ikke
    return X_amplitude, X_latency,X_ampl_late, X_diff,X_fft

def normalize_by_peak_latency(X):
    X = np.transpose(X)
    # Calculate the peak latency for each signal
    peak_latencies = np.argmax(X, axis=1)

    # Calculate the offset needed to shift the peak to index 20
    offset = 20 - peak_latencies

    # Shift each signal by the corresponding offset
    X_normalized = np.zeros(X.shape)
    for i, row in enumerate(X):
        X_normalized[i, :] = np.roll(row, offset[i])
    # taking the max distance to the peak(offset), and slice that number of elements. So it is ensured that it doesnt roll to the end of the signal. 
    X_normalized = X_normalized[:, :offset[20]] #-np.argmax(offset)
    X_normalized = np.transpose(X_normalized)
    return X_normalized

"""
Denne funktion normalisere inden for hvert subject
def normalize_X(X, groups):
    #normalized X
    subjects = np.unique(groups)
    X_norm = []
    # Normalize X for each subject
    for subj in subjects:
        # Extract rows for this subject
        X_subj = np.transpose(np.transpose(X)[groups == subj, :])
        X_subj_t = np.transpose(X_subj)


        # compute minimum and maximum values along columns
        max_abs = np.max(np.abs(X_subj_t)).reshape((-1, 1))
        min_abs = np.min(np.abs(X_subj_t)).reshape((-1, 1))
        new_max = 1


        # Scale the matrix using min-max normalization
        X_subject_norm = (X_subj / max_abs) * new_max
        
        #Concatenate here so X_norm get all the matrix from X_subject_norm
        X_norm.append(np.transpose(X_subject_norm))


    X_norm = np.transpose(np.vstack(X_norm))

    X_norm = normalize_by_peak_latency(X_norm)
    return X_norm
"""
"""Denne virker ikke helt, fandt vi ud af til m'det med krissi
def normalize_X(X, groups):
    #normalized X
    subjects = np.unique(groups)
    X_norm = []
    n_signals = X.shape[1]
    # Normalize X for each subject
    for signal in range(n_signals) :
        # Extract rows for this subject
        X_signal = np.transpose(X)[signal]
        


        # compute minimum and maximum values along columns
        max_abs = np.max(np.abs(X_signal)).reshape((-1, 1))
        min_abs = np.min(np.abs(X_signal)).reshape((-1, 1))
        new_max = 1


        # Scale the matrix using min-max normalization
        X_signal_norm = (X_signal / max_abs[0]) * new_max
        
        #Concatenate here so X_norm get all the matrix from X_subject_norm
        X_norm.append(np.transpose(X_signal_norm))
    
    signal_matrix = np.zeros((1583, 85))

    # stack the signals into the matrix
    for i, signal in enumerate(X_norm):
        signal_matrix[i, :] = signal
    X_norm_signals = np.transpose(signal_matrix)
    #X_norm = np.vstack(X_norm)

    X_norm = normalize_by_peak_latency(X_norm_signals)
    return X_norm"""

def normalize_X(X, groups):
    #normalized X
    subjects = np.unique(groups)
    X_norm = []
    n_signals = X.shape[1]
    # Normalize X for each subject
    for signal in range(n_signals):
        # extract the signal
        X_signal = X[:, signal]
        
        # compute the maximum and minimum values of the signal
        max_val = np.max(X_signal)
        min_val = np.min(X_signal)
        
        # add a small epsilon value to max_val to avoid division by zero
        epsilon = 1e-8
        max_val += epsilon
        
        # scale the signal using min-max normalization
        #X_signal_norm = (X_signal - min_val) / (max_val - min_val)
        X_signal_norm = 2 * (X_signal - np.min(X_signal)) / (np.max(X_signal) - np.min(X_signal)) - 1

        
        # append the normalized signal to X_norm
        X_norm.append(X_signal_norm)
        
    # stack the normalized signals into a matrix
    X_norm_signals = np.vstack(X_norm).T

    X_norm = normalize_by_peak_latency(X_norm_signals)
    return X_norm

def datapreprocess_tensor_transformer():
    main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"
    filelist = get_all_paths(main_path)
    X, y, groups, list_subjects = get_all_data(filelist)
    num_signals = X.shape[1]
    X = np.reshape(X, (num_signals, 85, 1))
    X = torch.from_numpy(X).float()
    #X = X.unsqueeze(1) # Add a channel dimension # (number of signals, number of input channels, signal length)
    y = torch.tensor(y).float()
    # Replace these variables with your actual data and model
    #subset_data = data[:5]
    groups_data = groups
    return X, y, groups_data

def datapreprocess_tensor():
    main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"
    filelist = get_all_paths(main_path)
    X, y, groups, list_subjects = get_all_data(filelist)
    num_signals = X.shape[1]
    X = np.reshape(X, (num_signals, 85, 1))
    X = torch.from_numpy(X).float()
    #X = X.unsqueeze(1) # Add a channel dimension # (number of signals, number of input channels, signal length)
    y = torch.tensor(y).float()
    # Replace these variables with your actual data and model
    #subset_data = data[:5]
    groups_data = groups
    return X, y, groups_data

def datapreprocess_tensor_cnn():
    main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"
    filelist = get_all_paths(main_path)
    X, y, groups, list_subjects = get_all_data(filelist)
    num_signals = X.shape[1]
    #https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    X = np.reshape(X, (num_signals, 1,85)) #Kristoffer
    X = torch.from_numpy(X).float()
    #X = X.unsqueeze(1) # Add a channel dimension # (number of signals, number of input channels, signal length)
    y = torch.tensor(y).float()
    # Replace these variables with your actual data and model
    #subset_data = data[:5]
    groups_data = groups
    return X, y, groups_data

if __name__ == "__main__":
    main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"
    filelist = get_all_paths(main_path)
    X, y, groups, list_subjects = get_all_data(filelist)

    X_norm = normalize_X(X, groups)
    plot_functions.plot_subject_coil(X_norm,y,list_subjects,groups,False,12)
    k = 1
    #X_amplitude, X_latency,X_ampl_late, X_diff,X_fft = data.other_X(X_norm)