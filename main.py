import matplotlib.pyplot as plt
import numpy as np
import data 
import models 
#import plot_functions
from pymatreader import read_mat
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense



main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"



filelist = data.get_all_paths(main_path)
#print(filelist)
X, y, groups, list_subjects = data.get_all_data(filelist)

#X_amplitude, X_latency,X_ampl_late, X_norm = data.other_X(X)

"""#How to plot
idx = np.where(np.array(groups) == 0)
X[idx]
y = np.array(y)
Xt =np.transpose(X)[idx]
Xtt = np.transpose(Xt)

idxy = np.array(idx)

PA = Xt[idx][np.where(y[idxy]==1)[1]]
AP =  Xt[idx][np.where(y[idxy]==2)[1]]

plt.plot(np.transpose(PA), color = "blue")
plt.plot(np.transpose(AP), color = "red")
plt.show()
m
"""


'''#print et enkelt eksempel
#path = "/mnt/projects/USS_MEP/COIL_ORIENTATION/x71487_coil_orient.mat"
path = "/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X99909_ses-1_task-coilorientation_emg.mat"
data1 = read_mat(path)
if "sub" in path:
    key = list(data1.keys())[3]
else:
    key = list(data1.keys())[0]

X_raw = data1[key]['values'][:,0]
y = data1[key]['frameinfo']['state']
X_raw, y = data.delete_frames(X_raw,y)
print("hey")
plt.plot(X_raw)
plt.show()
'''

def CNN(X, y, groups):
    X = np.array(np.transpose(X)) #np.transpose(
    y = np.array(y)
    groups = np.array(groups)

    tot_scores = []
    tot_indi_scores = []
    Coefficients = []
    intercepts = []
    coef_meanssss = []
    logo = LeaveOneGroupOut()
    scores = []

    # Create the CNN model
    # Define the input shape
    input_shape = (85, 1)

    # Define the number of filters and kernel size for the convolutional layer
    filters = 32
    kernel_size = 3

    model = Sequential()
    # Add a convolution layer with filters, kernel size, and ReLU activation
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    # Add a max pooling layer
    model.add(MaxPooling1D(pool_size=2))
    # Add another convolution layer with filters, kernel size, and ReLU activation
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    # Add another max pooling layer
    model.add(MaxPooling1D(pool_size=2))
    # Add a fully connected flattened layer
    model.add(Flatten())
    # Add a fully connected layer with 128 neurons and ReLU activation
    model.add(Dense(128, activation='relu'))
    # Add a fully connected layer with 2 neurons and softmax activation for binary classification
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model with categorical crossentropy loss and Adam optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for subject in set(groups):

        #logo.get_n_splits(X, y, list(set(groups)))
        scores = []

        #for train_index, test_index in logo.split(X[temp_subject], y[temp_subject], temp_subject[0]):
        temp_subject = np.where(groups == subject)
        test = list(range(0,len(temp_subject[0])))
        for train_index, test_index in skf.split(X[temp_subject[0]], y[temp_subject[0]]):
            #for train_index, test_index in loo.split(X[temp_subject[0]], y[temp_subject[0]],test):
            # select the indices among 1683 traj for subject
            X_train, X_test = X[temp_subject[0]][train_index], X[temp_subject[0]][test_index]
            y_train, y_test = y[temp_subject[0]][train_index], y[temp_subject[0]][test_index]
            
            #CNN model on each fold
            model.fit(X_train, y_train, epochs=10, verbose=0)
            # Evaluate the model on the test data
            X_test, y_test = X[test_index], y[test_index]
            X_test = np.reshape(X_test, (*X_test.shape, 1))
            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
            scores.append(accuracy)
            tot_scores.extend(scores)
        tot_indi_scores.append(scores)
        mean = []
        for i in tot_indi_scores:
            mean.append(np.mean(i))
        mean_indi_scores = mean
            

    return tot_scores, tot_indi_scores, mean_indi_scores

# Print the mean accuracy across all folds and subjects
#print('Mean accuracy:', np.mean(accs))




def barplot(X, groups, mean_indi_scores): #husk at tjek om onerow er slået til eller ej
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    axs.set_xlabel('Group number')
    axs.set_ylabel('Mean accuracy')
    axs.set_title('Barplot of mean accuracy trained on amplitude and latency')
    plt.bar(np.sort(list(set(groups))),mean_indi_scores)
    plt.show()


def PCA(X, explained = False, n=2, PCAs = True):# skal ind i plots
    from sklearn.decomposition import PCA
    if PCAs == True:
        # Create a PCA object with the desired number of components
        pca = PCA(n_components=n)

        # Fit the PCA model to the data and transform the data to the new space
        X_pca = pca.fit_transform(X)

        # The transformed data will now have two columns, which are the principal components
        # Create a scatter plot of the transformed data
        #plt.scatter(X_pca[:, 0], X_pca[:, 1])

        ''' # Add axis labels and a title
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Scatter Plot')

        # Show the plot
        plt.show()'''
        
        plt.plot(X_pca[:, 0], label = "PC1")
        plt.plot(X_pca[:, 1], label = "PC2")
        plt.legend(loc="upper left")
        plt.title('Principal components')
        plt.xlabel('Time')
        plt.ylabel('Value')
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


tot_scores, tot_indi_scores, mean_indi_scores = CNN(X, y, groups)

print(np.mean(tot_scores))
#X_amplitude, X_latency,X_ampl_late = data.other_X(X)
#tot_scores, tot_indi_scores, mean_indi_scores = loo_logisticregression_prsubject_stratified(X_norm, y, groups, onerow = False)

#print(np.mean(tot_scores))
#score, X_train, X_test, y_train, y_test, predictions = models.logregr(X_train, X_test, y_train, y_test)
#models.confmat(y_test, predictions, score)
#print(score)





