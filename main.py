import matplotlib.pyplot as plt
import numpy as np
import data 
import models 
#import plot_functions

main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"

"""
path_x01666 = data.path_x0166
path_x02299 = "/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X02299_ses-1_task-coilorientation_emg.mat"
X,y,X_sliced = data.get_one_data(path_x02299)

plt.plot(X_sliced)
plt.show()
"""

filelist = data.get_all_paths(main_path)
X, y, groups = data.get_all_data(filelist)

tot_scores, tot_indi_scores, mean_indi_scores = logo_logisticregression_prsubject(X, y, groups)

plt.bar(list(set(groups),mean)
plt.show()     

"""
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
X = np.array(np.transpose(X))
y = np.array(y)
groups = np.array(groups)

tot_scores = []
tot_indi_scores = []
for subject in set(groups):
    logo = LeaveOneGroupOut()
    #logo.get_n_splits(X, y, list(set(groups)))
    scores = []
    
    #for train_index, test_index in logo.split(X[temp_subject], y[temp_subject], temp_subject[0]):
    temp_subject = np.Xwhere(groups == subject)
    test = list(range(0,len(temp_subject[0])))
    for train_index, test_index in logo.split(X[temp_subject[0]], y[temp_subject[0]],test):
        #try:
        
        X_train, X_test = X[temp_subject[0]][train_index], X[temp_subject[0]][test_index]
        y_train, y_test = y[temp_subject[0]][train_index], y[temp_subject[0]][test_index]
        if len(list(set(y_train))) > 1:
            lr = LogisticRegression()
            lr.fit(X_train, y_train)
            accuracy = lr.score(X_test, y_test)
            scores.append(accuracy)
        else:
            print("smaller than 2", subject)
        #except:
        #    print("fail",subject)
    tot_scores.extend(scores)
    tot_indi_scores.append(scores)
print(np.mean(tot_scores), scores)
print("l[l[l[l]]]")

"""


#score, X_train, X_test, y_train, y_test, predictions = models.logregr(X_train, X_test, y_train, y_test)
#models.confmat(y_test, predictions, score)
#print(score)
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

