#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:35:32 2019

@author: becca

Bayesian ML Homework 2
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import *


# Analytics and modeling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Visualisation 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from pylab import savefig
import seaborn as sns; sns.set()

# Setting graphing preferences
sns.set(style="darkgrid", color_codes=True)

# Printing
import locale

# Show plots locally
locale.setlocale( locale.LC_ALL, '' )


#%% Problem 1

#%% Problem 2

#%% Problem 3

mushroom_data = (pd.read_csv("MushroomData.csv", header = 0)).astype('category').dropna(axis=0)

# Convert categorical data to binary values

mushroom_data = pd.get_dummies(mushroom_data, columns = list(set(mushroom_data.columns) - set(mushroom_data.EDIBLE)), drop_first = True)

mushroom_data["EDIBLE"] = np.where(mushroom_data["EDIBLE"] == "EDIBLE", 1, 0)

train_mushrooms, test_mushrooms = train_test_split(mushroom_data, test_size = .33, random_state = 20)

probabilityEdible = len(train_mushrooms[train_mushrooms.EDIBLE == 1]) / len(train_mushrooms)

probabilityPoisonous = 1 - probabilityEdible

# Fit binomial naive bayes classifier

y_train = train_mushrooms.EDIBLE

X_train = train_mushrooms.drop("EDIBLE", axis = 1)

y_test = test_mushrooms.EDIBLE

X_test = test_mushrooms.drop("EDIBLE", axis = 1)

mushroom_edible = BernoulliNB()

mushroom_edible.fit(X_train,y_train)

labels = mushroom_edible.predict(X_test)

# Metrics

classification_report(y_test,labels)
mat = confusion_matrix(y_test,labels)
roc_auc_score(y_test,labels)

# Visualisation

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=["Poisonous", "Edible"], yticklabels= ["Predicted Poisonous", "Predicted Edible"])
#%% Problem 4