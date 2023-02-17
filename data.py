import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from pymatreader import read_mat
import os
#path to all the files
main_path ="/mnt/projects/USS_MEP/COIL_ORIENTATION"
path_x01666 = "/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X01666_ses-1_task-coilorientation_emg.mat"

def get_all_paths(main_path):
    #we shall store all the file names in this list
    filelist = []


    #Making a list of all the files paths to all the data files. 
    for root, dirs, files in os.walk(main_path):
        for file in files:
                #append the file name to the list
            filelist.append(os.path.join(root,file))
    #Removes all the files that Lasse havent checked yet, so only using files with sub.
    filelist = [ x for x in filelist if "sub" in x ]
    return filelist

def get_one_data(path):
  data = read_mat(path)
  key = list(data.keys())[3]

  X = np.transpose(data[key]['values'][:,0])
  y = data[key]['frameinfo']['state']
  X_sliced = []
    #Vurder hvornår vi skal slice fra og til. 
  x_start, x_end = int(len(X[0,:])/2+20) , int(len(X[0,:])/2+200)
  for i in range(len(X)):
    X_sliced.append(X[i,:][x_start:x_end])
  X_sliced = np.transpose(np.array(X_sliced))
  #plt.plot(X_sliced)
  #plt.show()
  return X,y, X_sliced

def get_all_data(filelist):
    X = np.empty((20000,160))
    for path in filelist:
        data = read_mat(path)
        key = list(data.keys())[3]

        X, y, X_sliced = get_one_data(path)
        np.append(X_sliced, [X_temp])
        """
        X_temp = np.transpose(data[key]['values'][:,0])
        y_temp = data[key]['frameinfo']['state']
        X_sliced = []
        x_start, x_end = int(len(X[0,:])/2+20) , int(len(X[0,:])/2+200)
        for i in range(len(X_temp)):
            X.append(X_temp[i,:][x_start:x_end])
        #print(path)
        #np.concatenate((X,X_sliced), axis = 0)
        print(len(X))
        #getdata(i, key)
        """
        return X
    


filelist = get_all_paths(main_path)
X = get_all_data(filelist)
print(X)
