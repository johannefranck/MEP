import matplotlib.pyplot as plt
import numpy as np
import data 
import models 

main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"

"""
path_x01666 = data.path_x01666
path_x02299 = "/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X02299_ses-1_task-coilorientation_emg.mat"
X,y,X_sliced = data.get_one_data(path_x02299)

plt.plot(X_sliced)
plt.show()
"""

filelist = data.get_all_paths(main_path)
X, y, groups = data.get_all_data(filelist)
"""
from sklearn.model_selection import train_test_split, GroupKFold

# Split the data into training and testing sets based on groups
groups_train, groups_test = train_test_split(groups, test_size=0.2, random_state=42)
    
# Get the indices of the training groups
train_groups_idx = np.where(np.isin(groups, groups_train))[0]

# Define the training data
n_splits = int(len(set(groups_train)) * 0.8)  # 80% of the groups for training
group_kfold = GroupKFold(n_splits=n_splits)
train_indexes, _ = next(group_kfold.split(X[train_groups_idx], y[train_groups_idx], groups[train_groups_idx]))
X_train, y_train = X[train_groups_idx][train_indexes], y[train_groups_idx][train_indexes]

# Get the indices of the testing groups
test_groups_idx = np.where(np.isin(groups, groups_test))[0]

# Define the testing data
X_test, y_test = X[test_groups_idx], y[test_groups_idx]
"""
#data.train_test_split(X,y,groups)
#data.plotgroups(X, groups)


"""
PAlist = []
APlist = []

for i in range(len(y)):
    if y[i] == 1: #PA: 1
        PAlist.append(i) #index in y
        PAi = X[:,y[i]]
        PAs = PAlist.append(PAi)
    elif y[i] == 2: #AP: 2
        APlist.append(i)

PAmeans = []
APmeans = []
for i in range(len(PAlist)):
    PAi = X[:,PAlist[i]]
PAs = np.vstack([PAi])
    #PAmean = PAmeans.append(np.mean(PAs[i,:]))

plt.plot(X)
plt.show()
"""

#logregscore, X_train, X_test, y_train, y_test, predictions = models.logregr(np.transpose(X),y)
#models.confmat(y_test, predictions, logregscore)


#MLPscore, x_train, x_test, y_train, y_test, predictions, predictions_prob = models.MLP(X_sliced,y)
#models.confmat(y_test, predictions)

"""Noter
cut lm ud. 
Til resampling: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html
Træn på forskellen i amplitude, altse max -min
"""

