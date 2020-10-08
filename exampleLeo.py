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



# Loading some example data
iris = datasets.load_iris()
X = iris.data[:, :3]
y = iris.target

X = PCA(n_components=3).fit_transform(X)
print(X.shape)

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
#clf1 = RandomForestClassifier(random_state=0)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

xx1, yy1 = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))


xx2, zz2= np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(z_min, z_max, 0.1))

yy3, zz3= np.meshgrid(np.arange(y_min, y_max, 0.1),
                     np.arange(z_min, z_max, 0.1))


fig = plt.figure(1, figsize=(12, 12))

clf1.fit(X[:,[0,1]], y)
Z1 = clf1.predict(np.c_[xx1.ravel(), yy1.ravel()])

clf1.fit(X[:,[0,2]], y)
Z2 = clf1.predict(np.c_[xx2.ravel(), zz2.ravel()])

clf1.fit(X[:,[1,2]], y)
Z3 = clf1.predict(np.c_[yy3.ravel(), zz3.ravel()])

Z1 = Z1.reshape(xx1.shape)   
Z2 = Z2.reshape(xx2.shape)
Z3 = Z3.reshape(yy3.shape)

"""
print(Z1.shape)
print(xx1.shape)
print(yy1.shape)

print(Z2.shape)
print(xx2.shape)
print(zz2.shape)

print(Z3.shape)
print(yy3.shape)
print(zz3.shape)
"""



#grafico xy 3d
ax = fig.add_subplot(3, 2, 1, projection='3d')
ax.contour3D(xx1, yy1, Z1, 50, cmap='binary')
#ax.contour3D(xx2, Z2, zz2, 50, cmap='hot' )
#ax.contour3D(Z3, yy3, zz3, 50, cmap='cool')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
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

ax = fig.add_subplot(3, 2, 2)
ax.contourf(xx1, yy1, Z1, alpha=0.4)
ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
ax.set_xlabel("X")
ax.set_ylabel("Y")

#grafico xz 3d
ax = fig.add_subplot(3, 2, 3, projection='3d')
#ax.contour3D(xx1, yy1, Z1, 50,cmap='binary')
ax.contour3D(xx2, Z2, zz2, 50,colkey=TRUE, cmap='Accent' )
#ax.contour3D(Z3, yy3, zz3, 50, cmap='cool')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.view_init(15, -90)
ax.set_title("XZ")
ax.set_xlabel("X")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Y")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Z")
ax.w_zaxis.set_ticklabels([])

#grafico xz 2d
ax = fig.add_subplot(3, 2, 4)
ax.contourf(xx2, zz2, Z2, alpha=0.4)
ax.scatter(X[:, 0], X[:, 2], c=y, s=20, edgecolor='k')
ax.set_xlabel("X")
ax.set_ylabel("Z")

#grafico yz 3d
ax = fig.add_subplot(3, 2, 5, projection='3d')
#ax.contour3D(xx1, yy1, Z1, 50,cmap='binary')
#ax.contour3D(xx2, Z2, zz2, 50, cmap='hot' )
ax.contour3D(Z3, yy3, zz3, 50, cmap='cool')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.view_init(195, 0)
ax.set_title("YZ")
ax.set_xlabel("X")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Y")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Z")
ax.w_zaxis.set_ticklabels([])

#grafico yz 2d
ax = fig.add_subplot(3, 2, 6)
ax.contourf(yy3, zz3, Z3, alpha=0.4)
ax.scatter(X[:, 1], X[:, 2], c=y, s=20, edgecolor='k')
ax.set_xlabel("Y")
ax.set_ylabel("Z")


plt.show()