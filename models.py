import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from pymatreader import read_mat
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression

#import data

#path_x01666 = data.path_x01666


#### Logistic regression model
def logregr(X_train, X_test, y_train, y_test):

  from sklearn.model_selection import LeaveOneGroupOut
  from sklearn.linear_model import LogisticRegression
  from sklearn.model_selection import cross_val_score
  #X = np.array(np.transpose(X))
  #y = np.array(y)
  #groups = np.array(groups)

  group_kfold = LeaveOneGroupOut()
  model = LogisticRegression()
  #model.split(X, y, groups)
  cvs = cross_val_score(model, X_train, y_train, cv=group_kfold)
  print(cvs)

  """
    group_kfold = LeaveOneGroupOut()
    print(group_kfold.get_n_splits(X, y, groups))
    #print(group_kfold)
    #GroupKFold(n_splits=2)
    scores = []
    for train_index, test_index in group_kfold.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        accuracy = lr.score(X_test, y_test)
        scores.append(accuracy)
        print(accuracy, scores)
    """


  """
  from sklearn.linear_model import LogisticRegression
  # all parameters not specified are set to their defaults
  logisticRegr = LogisticRegression(max_iter=2000)
  logisticRegr.fit(X_train, y_train)
  # Returns a NumPy Array
  # Predict for One Observation (image)
  logisticRegr.predict(X_test[0].reshape(1,-1))
  predictions = logisticRegr.predict(X_test)
  # Use score method to get accuracy of model
  score = logisticRegr.score(X_test, y_test)
  """

  return score, X_train, X_test, y_train, y_test, predictions

def confmat(y_test, predictions, score):
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
  plt.imshow(cm, interpolation='nearest', cmap='RdBu')
  plt.title('Accuracy Score: {0}'.format(score), size = 15)
  plt.colorbar()
  tick_marks = np.arange(2)
  plt.xticks(tick_marks, ["1", "2"], rotation=45, size = 10)
  plt.yticks(tick_marks, ["1", "2"], size = 10)
  plt.tight_layout()
  plt.ylabel('Actual label', size = 2)
  plt.xlabel('Predicted label', size = 2)
  width, height = cm.shape
  for x in range(width):
    for y in range(height):
      plt.annotate(str(cm[x][y]), xy=(y, x), 
      horizontalalignment='center',
      verticalalignment='center')
  plt.show()

#### MLP classifier
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
  

def logo_logisticregression_prsubject(X, y, groups, onerow = False):
  
  #Checking whether it is onerow, ex. if it is only apmlitude
  if onerow == True:
    X = np.array(X)
  else:
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
    temp_subject = np.where(groups == subject)
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
    mean = []
    for i in tot_indi_scores:
      mean.append(np.mean(i))
    mean_indi_scores = mean
  return tot_scores, tot_indi_scores, mean_indi_scores

#Run models
#X,y,X_sliced = data.get_one_data(path_x01666)

#logregscore, x_train, x_test, y_train, y_test, predictions = logregr(X_sliced,y)
#print(f"logregression score: {logregscore}")
#MLPscore, x_train, x_test, y_train, y_test, predictions, predictions_prob = MLP(X_sliced,y)
#print(f"MLP score: {MLPscore}")
