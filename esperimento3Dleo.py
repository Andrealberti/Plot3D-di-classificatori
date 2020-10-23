from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection



# Loading some example data
iris = datasets.load_iris()
X = iris.data
y = iris.target

X = PCA(n_components=3).fit_transform(X)


# Training classifiers
clf = DecisionTreeClassifier(max_depth=4)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

lista1=np.arange(3, 6, 1)
print("lista 1 ", lista1)
lista2=np.arange(-3, 0, 1)
print("lista 2 ", lista2)
lista3=np.arange(0, 2, 1)
print("lista 3 ", lista3)

listagrid1, listagrid2, listagrid3 = np.meshgrid(lista1,lista2,lista3)
print("lista grid1 ",listagrid1)
print("lista grid2 ",listagrid2)
print("lista grid3 ",listagrid3)
print("chimamataaaaa ",listagrid1[0,2])
print("chimamataaaaa ",listagrid2[0,2])
print("chimamataaaaa ",listagrid3[0,2])


xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1),
                         np.arange(z_min, z_max, 0.1))

clf.fit(X, y)
Z=clf.predict(np.c_[xx.ravel(), yy.ravel(),zz.ravel()])

Z=Z.reshape(xx.shape)

print("")

fig = plt.figure(1, figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, projection='3d')

x = np.arange(x_min, x_max, 0.1)
y = np.arange(y_min, y_max, 0.1),
z = np.arange(z_min, z_max, 0.1),





ax.scatter(xx[:10,:10,:10],yy[:10,:10,:10],zz[:10,:10,:10], alpha=0.4, c=Z[:10,:10,:10], s=20, edgecolor='k')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=20, edgecolor='k')
plt.show()


"""

fig = plt.figure(1, figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.contour3D(xx, yy, zz, 3,  s=20, edgecolor='k', alpha=0.4)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=20, edgecolor='k')
"""

"""



fig = plt.figure(1, figsize=(12, 12))

Z1 = Z1.reshape(xx1.shape)   
Z2 = Z2.reshape(xx2.shape)
Z3 = Z3.reshape(yy3.shape)


#grafico xy 3d
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.contourf(xx1, yy1, Z1, 3,  s=20, edgecolor='k', alpha=0.4)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=20, edgecolor='k')
#imposto l'angolo di rotazione iniziale
ax.view_init(105, -90)

ax.set_title("XY")
ax.set_xlabel("X")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Y")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Z")
ax.w_zaxis.set_ticklabels([])

#grafico xy 2d

ax = fig.add_subplot(1, 2, 2)
ax.contourf(xx1, yy1, Z1, alpha=0.4)
ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
ax.set_xlabel("X")
ax.set_ylabel("Y")


plt.show()
"""