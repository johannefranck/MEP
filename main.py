import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from pymatreader import read_mat
import os
import data
import models



path_x01666 = data.path_x01666
X,y,X_sliced = data.get_one_data(path_x01666)

logregscore, x_train, x_test, y_train, y_test, predictions = models.logregr(X_sliced,y)
models.confmat(y_test, predictions)

MLPscore, x_train, x_test, y_train, y_test, predictions, predictions_prob = models.MLP(X_sliced,y)
models.confmat(y_test, predictions)


