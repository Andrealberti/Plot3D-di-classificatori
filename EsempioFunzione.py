import numpy as np
from FunctionPlot import PlotConvexHull3D, plotDecision2D

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



#chiamiamo la funzione passando i parametri richiesti
#prendo il mio drataset che passerò ai metodi
TrainingSet = datasets.load_iris()

#definiso il mio classificatore
#clf = DecisionTreeClassifier(max_depth=4)
clf = RandomForestClassifier(random_state=0)


#plotDecision2D(TrainingSet, clf, True)

PlotConvexHull3D(TrainingSet,clf,0.2)
