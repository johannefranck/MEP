import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from pymatreader import read_mat
import os
path ="/mnt/projects/USS_MEP/COIL_ORIENTATION"
#we shall store all the file names in this list
filelist = []

for root, dirs, files in os.walk(path):
  for file in files:
        #append the file name to the list
    filelist.append(os.path.join(root,file))
  #print(f"filelist {filelist}")
#Removes all the files that Lasse havent checked yet, so only using files with sub.
filelist = [ x for x in filelist if "sub" in x ]
"""
X = np.empty((20000,160))
for path in filelist:
  data = read_mat(path)
  key = list(data.keys())[3]

  X_temp = np.transpose(data[key]['values'][:,0])
  y_temp = data[key]['frameinfo']['state']
  X_sliced = []
  for i in range(len(X_temp)):
    X_sliced.append(X_temp[i,:][10040:10200])
  #print(path)
  #np.concatenate((X,X_sliced), axis = 0)
print(len(X))
  #getdata(i, key)
#matlab_data = sio.loadmat('/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X01666_ses-1_task-coilorientation_emg.mat',verify_compressed_data_integrity = False)
#matlab_data = matlab_data['x01666_coilorient000_1_wave_data']

"""
def getdata(path):
  data = read_mat(path)
  key = list(data.keys())[3]
  #plt.plot(data[key]['values'][:,0].mean(-1))
  X = np.transpose(data[key]['values'][:,0])#.mean(-1)
  y = data[key]['frameinfo']['state']
  X_sliced = []

  x_start, x_end = int(len(X[0,:])/2+40) , int(len(X[0,:])/2+200)
  for i in range(len(X)):
    X_sliced.append(X[i,:][x_start:x_end])
    #print(X_sliced)
  X_sliced = np.transpose(np.array(X_sliced))
  plt.plot(X_sliced)
  plt.show()
  return X,y, X_sliced




######Logistic regression model
def logregr(X,y):
  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=0)
  from sklearn.linear_model import LogisticRegression
  # all parameters not specified are set to their defaults
  logisticRegr = LogisticRegression(max_iter=2000)
  logisticRegr.fit(x_train, y_train)
  # Returns a NumPy Array
  # Predict for One Observation (image)
  logisticRegr.predict(x_test[0].reshape(1,-1))
  predictions = logisticRegr.predict(x_test)
  # Use score method to get accuracy of model
  score = logisticRegr.score(x_test, y_test)
  return score, x_train, x_test, y_train, y_test, predictions

def confmat(y_test, predictions):
  cm = metrics.confusion_matrix(y_test, predictions)
  print(cm)
  """
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
  plt.ylabel('Actual label');
  plt.xlabel('Predicted label');
  all_sample_title = 'Accuracy Score: {0}'.format(score)
  plt.title(all_sample_title, size = 15);
  """



  plt.figure(figsize=(5,5))
  plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
  plt.title('Accuracy Score: {0}'.format(score), size = 15)
  plt.colorbar()
  tick_marks = np.arange(6)
  plt.xticks(tick_marks, ["1", "2", "3", "4", "5", "6"], rotation=45, size = 10)
  plt.yticks(tick_marks, ["1", "2", "3", "4", "5", "6"], size = 10)
  plt.tight_layout()
  plt.ylabel('Actual label', size = 6)
  plt.xlabel('Predicted label', size = 6)
  width, height = cm.shape
  for x in range(width):
    for y in range(height):
      plt.annotate(str(cm[x][y]), xy=(y, x), 
      horizontalalignment='center',
      verticalalignment='center')
  plt.show()



####
path = '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X36523_ses-1_task-coilorientation_emg.mat'
X,y, X_sliced = getdata(path)
#plt.plot(data[key]['values'][:,0].mean(-1))

#score, x_train, x_test, y_train, y_test, predictions = logregr(X_sliced,y)
#print(f"score: {score}")
#confmat(y_test, predictions)
