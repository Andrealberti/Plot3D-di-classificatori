import sys

print(sys.version)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions

from mpl_toolkits import mplot3d    #L aggiungo libreria per il 3d
from sklearn import datasets #L
from sklearn.decomposition import PCA #L

# Initializing Classifiers
clf1 = LogisticRegression(random_state=0)
clf2 = RandomForestClassifier(random_state=0)
clf3 = SVC(random_state=0, probability=True)
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[2, 1, 1], voting='soft')

#L

# Loading some example data
#iris = iris_data()
#print(iris)
#X, y = iris_data()
#X = X[:,[0, 3]]
#

Xx, y = iris_data()
X = Xx[:,[0, 3]]
Z = Xx[:,[1, 2]]




"""
#L
iris = datasets.load_iris()
X = iris.data[:, :2]  
y = iris.target
"""


#

# Plotting Decision Regions

gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10, 8))

labels = ['Logistic Regression',
          'Random Forest',
          'RBF kernel SVM',
          'Ensemble']

for clf, lab, grd in zip([clf1, clf2, clf3, eclf],
                         labels,
                         itertools.product([0, 1],
                         repeat=2)):
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])

    fig = plt.figure(1, figsize=(8, 6))

    fig = plot_decision_regions(X=X, y=y,
                                clf=clf, legend=2)
    plt.title(lab)

plt.show()
