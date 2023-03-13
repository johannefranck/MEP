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


#logregscore, X_train, X_test, y_train, y_test, predictions = models.logregr(np.transpose(X),y)
#models.confmat(y_test, predictions, logregscore)


#MLPscore, x_train, x_test, y_train, y_test, predictions, predictions_prob = models.MLP(X_sliced,y)
#models.confmat(y_test, predictions)

"""Noter
cut lm ud. 
Til resampling: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html
Træn på forskellen i amplitude, altse max -min
"""

