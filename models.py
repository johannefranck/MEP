import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from pymatreader import read_mat
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def MLP(X,y):
  from sklearn.neural_network import MLPClassifier
  from sklearn.model_selection import train_test_split
  #X, y = make_classification(n_samples=100, random_state=1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
  clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
  predictions_prob = clf.predict_proba(X_test[:1])
  predictions = clf.predict(X_test[:5, :])
  score = clf.score(X_test, y_test)
  return score, X_train, X_test, y_train, y_test, predictions, predictions_prob
  

def kfold_logisticregression_prsubject_stratified(X, y, groups, onerow = False):
    """
    Logistic Regression model doing 10-fold cross validation (stratified).
    Trains one model pr subject
    """
    #Checking whether it is onerow, one feature ie. if it is only apmlitude or latency
    if onerow == True:
        X = np.array(X)
    else:
        X = np.array(np.transpose(X))
    y = np.array(y)
    groups = np.array(groups)

    tot_scores = []
    tot_indi_scores = []
    all_subject_coefficients = []

    for subject in set(groups): #{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ...}
        scores = []
        subject_coefficients = []
        temp_subject = np.where(groups == subject)# the subjects index's
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(X[temp_subject[0]], y[temp_subject[0]]):
            X_train, X_test = X[temp_subject[0]][train_index], X[temp_subject[0]][test_index]
            y_train, y_test = y[temp_subject[0]][train_index], y[temp_subject[0]][test_index]

            lr = LogisticRegression()
            lr.fit(X_train, y_train)
            accuracy = lr.score(X_test, y_test)
            coef = lr.coef_
            subject_coefficients.append(coef[0])
            scores.append(accuracy)
            tot_scores.extend(scores)
        # Average the coefficients over all folds for this subject
        subject_coefficients_mean = np.mean(subject_coefficients, axis=0)
        all_subject_coefficients.append(subject_coefficients_mean)
        tot_indi_scores.append(scores)
        mean = []
        for i in tot_indi_scores:
            mean.append(np.mean(i))
        mean_indi_scores = mean
    return tot_scores, tot_indi_scores, mean_indi_scores, all_subject_coefficients

def k10fold_logreg_generel_model(X, y, onerow =False):
    """ 
    Logistic Regression model doing 10-fold cross validation (stratified).
    Trains one model on all trajectories shuffled
    To train on normalized data, input Xnorm
    Returns the accuracy for the 10 folds, and the mean score, and the coefficient weight vector
    """

    if onerow == True:
        X = np.array(X)
    else:
        X = np.array(np.transpose(X))
    y = np.array(y)

    scores = []
    coefficients = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        coef = lr.coef_
        coefficients.append(coef[0])
        accuracy = lr.score(X_test, y_test)
        scores.append(accuracy)

    return scores, np.mean(scores), coefficients

def kfold_svm_prsubject_stratified(X, y, groups, onerow = False):
    """
    SVM model doing 10-fold cross validation (stratified).
    Trains one model pr subject
    """
    #Checking whether it is onerow, one feature ie. if it is only apmlitude or latency
    if onerow == True:
        X = np.array(X)
    else:
        X = np.array(np.transpose(X))
    y = np.array(y)
    groups = np.array(groups)

    tot_scores = []
    tot_indi_scores = []
    all_subject_coefficients = []

    for subject in set(groups): #{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ...}
        scores = []
        subject_coefficients = []
        temp_subject = np.where(groups == subject)# the subjects index's
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        for train_index, test_index in skf.split(X[temp_subject[0]], y[temp_subject[0]]):
            X_train, X_test = X[temp_subject[0]][train_index], X[temp_subject[0]][test_index]
            y_train, y_test = y[temp_subject[0]][train_index], y[temp_subject[0]][test_index]
        
            # Create an SVM classifier with a linear kernel
            svm = SVC(kernel='linear')

            # Train the classifier on the training data
            svm.fit(X_train, y_train)
            accuracy = svm.score(X_test, y_test)
            coef = svm.coef_
            subject_coefficients.append(coef[0])
            scores.append(accuracy)
            tot_scores.extend(scores)
        # Average the coefficients over all folds for this subject
        subject_coefficients_mean = np.mean(subject_coefficients, axis=0)
        all_subject_coefficients.append(subject_coefficients_mean)
        tot_indi_scores.append(scores)
        mean = []
        for i in tot_indi_scores:
            mean.append(np.mean(i))
        mean_indi_scores = mean
    return tot_scores, tot_indi_scores, mean_indi_scores, all_subject_coefficients

def k10fold_svm_generel_model(X, y, onerow =False):
    """ 
    SVM doing 10-fold cross validation (stratified).
    Trains one model on all trajectories shuffled
    To train on normalized data, input Xnorm
    Returns the accuracy for the 10 folds, and the mean score, and the coefficient weight vector
    """

    if onerow == True:
        X = np.array(X)
    else:
        X = np.array(np.transpose(X))
    y = np.array(y)

    scores = []
    coefficients = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        coef = lr.coef_
        coefficients.append(coef[0])
        accuracy = lr.score(X_test, y_test)
        scores.append(accuracy)

    return scores, np.mean(scores), coefficients


if __name__ == "__main__":
    import data
    import plot_functions
    main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"
    filelist = data.get_all_paths(main_path)
    X, y, groups, list_subjects = data.get_all_data(filelist) 
    # 10-fold stratified cross validation PR SUBJECT
    #tot_scores, tot_indi_scores, mean_indi_scores, all_subject_coefficients = kfold_svm_prsubject_stratified(X, y, groups, onerow = False)
    #print(np.mean(tot_scores))
    #print(f"number of subjects: {len(tot_indi_scores)}")
    #tot_scores_mean = np.mean(tot_scores)
    #plot_functions.barplot(groups, mean_indi_scores, acc = tot_scores_mean, xtype_title = 'X')

    STAA = 12.5  # sliced_time_after_artifact
    n_data_points = X.shape[0]
    time = np.arange(n_data_points) / 2000 * 1000 + STAA
#
    #plt.title('Mean weight vector over all subjects, X')
    #plt.plot(time,np.mean(all_subject_coefficients, axis=0))

    scores, mean_scores, coefficients = k10fold_svm_generel_model(X, y, onerow =False)
    plt.title('Mean weight vector over 10 folds, X')
    plt.xlabel('Time (ms)')
    plt.plot(time, np.mean(coefficients,axis=0))
    plt.show()
