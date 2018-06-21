# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 22:37:46 2018

@author: adity
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn

data = pd.read_csv('creditcard.csv')
data = data.sample(frac = 0.1, random_state =1)

fraud = data[data['Class']==1]
valid = data[data['Class']==0]

outlier_fraction = len(fraud)/float(len(valid))

#Correation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmat, vmax=0.8, square = True)

# Get columns from dataframe
columns = data.columns.tolist()

# Filter columns
columns = [c for c in columns if c not in ["Class"]]

# Store the variable to predict

target = "Class"
X= data[columns]
y= data[target]

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# define a random state
state=1

#outlier detection methods
classifiers = {
        "Isolation Forest": IsolationForest(max_samples = len(X),
                                            contamination = outlier_fraction,
                                            random_state = state),
        "Local Outlier Factor": LocalOutlierFactor(
                n_neighbors =20,
                contamination = outlier_fraction
                )
}

# Fit the model
n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate (classifiers.items()):
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    # Reshape the prediction values to 0 for valid, 1 for fraud
    y_pred[y_pred==1] = 0
    y_pred[y_pred==-1] = 1
    
    n_errors = (y_pred!=y).sum()
    
    # Run classification metricss
    print('{}:{}'.format(clf_name, n_errors))
    print(accuracy_score(y,y_pred))
    print(classification_report(y,y_pred))