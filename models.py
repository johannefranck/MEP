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
    THE RIGHT ONE! Logistic Regression model doing 10-fold cross validation (stratified).
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
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)

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

"""def k10fold_logreg_generel_model(X, y, onerow =False):
     
    WRONG
    Logistic Regression model doing 10-fold cross validation (stratified).
    Trains one model on all trajectories shuffled
    To train on normalized data, input Xnorm
    Returns the accuracy for the 10 folds, and the mean score, and the coefficient weight vector
    

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
"""

from sklearn.model_selection import LeaveOneGroupOut

def logo_logreg_model(X, y, groups, onerow =False):
    """ 
    THE RIGHT ONE. Logistic Regression model doing Leave-One-Group-Out cross validation.
    Trains one model on all trajectories shuffled
    To train on normalized data, input Xnorm
    Returns the accuracy for each fold, and the mean score, and the coefficient weight vector
    """

    if onerow == True:
        X = np.array(X)
    else:
        X = np.array(np.transpose(X))
    y = np.array(y)
    logo = LeaveOneGroupOut()
    scores = []
    coefficients = []

    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        coef = lr.coef_
        coefficients.append(coef[0])
        accuracy = lr.score(X_test, y_test)
        scores.append(accuracy)
    mean_score = np.mean(scores)
    return scores, mean_score, coefficients
'''
def kfold_logisticregression_prsubject_stratified(X, y, groups, onerow = False):#LOSO leave one subject out
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

    """tot_scores = []
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
        #mean_indi_scores = meanKfold(results of X, Xnorm, Xlatency, Xamplitude)
    else:
        X = np.array(np.transpose(X))
    y = np.array(y)"""

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
'''

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

def permute_within_groups(y, groups):
    #np.random.seed(10)
    y = np.array(y)
    permuted_y = y.copy()
    unique_groups = np.unique(groups)
    
    for group in unique_groups:
        group_indices = np.where(groups == group)[0]
        group_labels = y[group_indices]
        permuted_group_labels = np.random.permutation(group_labels)
        permuted_y[group_indices] = permuted_group_labels
    return permuted_y.tolist()

def perm_test_logo_logreg(X, y, groups, n_permutations=1000, onerow = False):
    """
    Conduct a permutation test using Leave-One-Group-Out cross-validation.
    """
    
    # Compute the actual scores
    actual_scores, actual_mean_score, _ = logo_logreg_model(X, y, groups, onerow = onerow)
    
    permuted_scores = []

    for i in range(n_permutations):
        # Permute the labels within each group
        permuted_y = permute_within_groups(y, groups)
        
        # Compute the permuted scores
        permuted_score, permuted_mean_score, _ = logo_logreg_model(X, permuted_y, groups, onerow = onerow)
        
        #permuted scores is a list of scores for each subjects
        permuted_scores.append(permuted_score)

        if i == 500:
            print("hey i am halfwat through")
    
    # Compute the p-value as the proportion of permuted mean scores
    # that are greater than or equal to the actual mean score
    p_value = np.mean(np.array(permuted_scores) >= actual_scores)
    permuted_mean_score = np.mean(permuted_scores)
    
    return p_value, permuted_scores,permuted_mean_score,actual_scores, actual_mean_score

def perm_test_kfold_prsubject(X, y, groups, n_permutations=1000, onerow = False):
    """
    Conduct a permutation test using kfold cross-validation.
    """
    
    # Compute the actual scores
    _, _, actual_scores, _ = kfold_logisticregression_prsubject_stratified(X, y, groups, onerow = onerow)
    actual_mean_score = np.mean(actual_scores)
    permuted_scores_list = []

    for i in range(n_permutations):
        # Permute the labels within each group
        permuted_y = permute_within_groups(y, groups)
        
        # Compute the permuted scores
        _, tot_indi_scores, permuted_scores, _ = kfold_logisticregression_prsubject_stratified(X, permuted_y, groups, onerow = onerow)
        permuted_mean_score = np.mean(permuted_scores)
        
        #permuted scores is a list of scores for each subjects
        permuted_scores_list.append(permuted_scores)

        if i == 500:
            print("hey i am halfway through")
    
    # Compute the p-value as the proportion of permuted mean scores
    # that are greater than or equal to the actual mean score
    p_value = np.mean(np.array(permuted_scores) >= actual_scores)
    permuted_mean_score = np.mean(permuted_scores)
    
    return p_value, permuted_scores,permuted_mean_score,actual_scores, actual_mean_score






if __name__ == "__main__":
    import data
    import plot_functions
    main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"
    filelist = data.get_all_paths(main_path)
    X, y, groups, list_subjects = data.get_all_data(filelist) 
    X_norm = data.normalize_X(X, groups)
    X_amplitude, X_latency,X_ampl_late, X_diff,X_fft = data.other_X(X)
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

    #scores, mean_scores, coefficients = k10fold_svm_generel_model(X, y, onerow =False)
    #scores, mean_scores, coefficients = kfold_logisticregression_prsubject_stratified(X, y,groups,  onerow =False)
    #tot_scores, tot_indi_scores, mean_indi_scores, all_subject_coefficients = kfold_logisticregression_prsubject_stratified(X, y, groups, onerow = False)


    #p_value, permuted_scores,permuted_mean_score,actual_scores, actual_mean_score = perm_test_logo_logreg(X, y, groups, n_permutations=1000)

    scores, mean_score, coefficients = logo_logreg_model(X, y, groups, onerow =False)
    tot_scores, tot_indi_scores, mean_indi_scores, all_subject_coefficients = kfold_logisticregression_prsubject_stratified(X, y, groups, onerow = False)
    p_value, permuted_scores,permuted_mean_score,actual_scores, actual_mean_score =perm_test_kfold_prsubject(X, y, groups, n_permutations=1000, onerow = False)
    print("yo this is for x ")
    print(f"permuted score {permuted_mean_score}")
    print(f"actual score {actual_mean_score}")
    print(f"p value: {p_value}")

    #p_value, permuted_scores,permuted_mean_score,actual_scores, actual_mean_score = perm_test_kfold_prsubject(X_norm, y, groups, n_permutations=1000, onerow = False)
    #print("yo this is for x norm")
    #print(f"permuted score {permuted_mean_score}")
    #print(f"actual score {actual_mean_score}")
    #print(f"p value: {p_value}")

    p_value, permuted_scores,permuted_mean_score,actual_scores, actual_mean_score =perm_test_kfold_prsubject(X_amplitude, y, groups, n_permutations=1000, onerow = True)
    print("yo this is for x amplitude")
    print(f"permuted score {permuted_mean_score}")
    print(f"actual score {actual_mean_score}")
    print(f"p value: {p_value}")

    p_value, permuted_scores,permuted_mean_score,actual_scores, actual_mean_score =perm_test_kfold_prsubject(X_latency, y, groups, n_permutations=1000, onerow = True)
    print("yo this is for x latency")
    print(f"permuted score {permuted_mean_score}")
    print(f"actual score {actual_mean_score}")
    print(f"p value: {p_value}")


    """plt.title('Mean weight vector over 10 folds, X')
    plt.xlabel('Time (ms)')
    plt.plot(time, np.mean(coefficients,axis=0))
    plt.show()
    """