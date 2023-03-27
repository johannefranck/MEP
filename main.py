import matplotlib.pyplot as plt
import numpy as np
import data 
import models 
#import plot_functions
from pymatreader import read_mat


main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"

"""
path_x01666 = data.path_x0166
path_x02299 = "/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X02299_ses-1_task-coilorientation_emg.mat"
X,y,X_sliced = data.get_one_data(path_x02299)

plt.plot(X_sliced)
plt.show()
"""

filelist = data.get_all_paths(main_path)
#print(filelist)
X, y, groups, list_subjects = data.get_all_data(filelist)

"""
print et enkelt eksempel
path = "/mnt/projects/USS_MEP/COIL_ORIENTATION/x71487_coil_orient.mat"
data1 = read_mat(path)
if "sub" in path:
    key = list(data1.keys())[3]
else:
    key = list(data1.keys())[0]

X_raw = data1[key]['values'][:,0]
y = data1[key]['frameinfo']['state']
X_raw, y = data.delete_frames(X_raw,y)
print("hey")
"""
X_amplitude, X_latency,X_ampl_late = data.other_X(X)
tot_scores, tot_indi_scores, mean_indi_scores = models.logo_logisticregression_prsubject(X, y, groups, onerow = False, LR = False, SVM = True)
print(np.mean(tot_scores))

# Perform SVD on the X matrix
#U, s, Vt = np.linalg.svd(X)

# Plot the singular values in a bar plot
#plt.bar(range(len(s)), s)
#plt.title("Singular Values")
#plt.xlabel("Index")
#plt.ylabel("Value")
#plt.show()


def barplot(X, groups, mean_indi_scores): #husk at tjek om onerow er slået til eller ej
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    axs.set_xlabel('Group number')
    axs.set_ylabel('Mean accuracy')
    axs.set_title('Barplot of mean accuracy trained on amplitude and latency')
    plt.bar(np.sort(list(set(groups))),mean_indi_scores)
    plt.show()


def PCA(X, explained = False, n=2, PCA = True):# skal ind i plots
    from sklearn.decomposition import PCA
    if PCA == True:
        # Create a PCA object with the desired number of components
        pca = PCA(n_components=n)

        # Fit the PCA model to the data and transform the data to the new space
        X_pca = pca.fit_transform(X)

        # The transformed data will now have two columns, which are the principal components
        # Create a scatter plot of the transformed data
        plt.scatter(X_pca[:, 0], X_pca[:, 1])

        # Add axis labels and a title
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Scatter Plot')

        # Show the plot
        plt.show()
    if explained == True:
        #Fit the PCA model to the data
        pca.fit(X)

        # Get the explained variance ratios
        variance_ratios = pca.explained_variance_ratio_

        # Create a bar plot of the explained variance ratios
        plt.bar(range(len(variance_ratios)), variance_ratios)

        # Add axis labels and a title
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Ratio by Principal Component')

        # Show the plot
        plt.show()



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

