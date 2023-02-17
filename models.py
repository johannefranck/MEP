import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from pymatreader import read_mat
import data

path_x01666 = data.path_x01666

#### Logistic regression model
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
  
X,y,X_sliced = data.get_one_data(path_x01666)


#logregscore, x_train, x_test, y_train, y_test, predictions = logregr(X_sliced,y)
#print(f"logregression score: {logregscore}")
#print("hi")

MLPscore, x_train, x_test, y_train, y_test, predictions, predictions_prob = MLP(X_sliced,y)
print(f"MLP score: {MLPscore}")
