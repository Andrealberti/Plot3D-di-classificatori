import numpy as np
from FunctionPlot import PlotConvexHull3D, plotDecision2D

from itertools import product

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier




#chiamiamo la funzione passando i parametri richiesti
#prendo il mio drataset che passerò ai metodi
TrainingSet = datasets.load_iris()

#definiso il mio classificatore
clf = DecisionTreeClassifier(max_depth=4)
clf = RandomForestClassifier(random_state=0)

#dichiaro una variavile che se positiva oltre al plot 2d lo riporta in 3d, ma sempre per classificatori 2d
DDD=False
#plotDecision2D(TrainingSet, clf, DDD)


PlotConvexHull3D(TrainingSet,clf)
