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


# Loading some example data
iris = datasets.load_iris()
X = iris.data[:, :3]
y = iris.target

X = PCA(n_components=3).fit_transform(X)


# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)

#ATTENZIONE HO CAMBIATO DIMENSIONE ARRAY PER IL PREDICT


# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1


xx1, yy1= np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

xx2, zz2= np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(z_min, z_max, 0.1))

yy3, zz3= np.meshgrid(np.arange(y_min, y_max, 0.1),
                     np.arange(z_min, z_max, 0.1))

#f = plt.subplot(2, 2, sharex='col', sharey='row', figsize=(10, 8))
fig = plt.figure(1, figsize=(10, 8))

clf1.fit(X[:,[0,1]], y)
Z1 = clf1.predict(np.c_[xx1.ravel(), yy1.ravel()])

clf1.fit(X[:,[0,2]], y)
Z2 = clf1.predict(np.c_[xx2.ravel(), zz2.ravel()])

clf1.fit(X[:,[1,2]], y)
Z3 = clf1.predict(np.c_[yy3.ravel(), zz3.ravel()])

Z1 = Z1.reshape(xx1.shape)   
Z2 = Z2.reshape(xx2.shape)
Z3 = Z3.reshape(yy3.shape)

#prov=np.zeros((len(Z1),1))
#ax = plt.axes(projection='3d')
ax = fig.add_subplot(2, 2, 1, projection='3d')
#ax.contour3D(xx1, yy1, Z1, 50,cmap='binary')
#ax.contour3D(xx2, zz2, Z2, 50, cmap='hot' )
ax.contour3D(yy3, zz3, Z3, 50, cmap='cool')

print(len(Z1))
print(len(Z2))
print(len(Z3))

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("X")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Y")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Z")
ax.w_zaxis.set_ticklabels([])

#grafico 2

ax = fig.add_subplot(2, 2, 2)
ax.contourf(xx1, yy1, Z1, alpha=0.4)
ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
ax.set_xlabel("X")
ax.set_ylabel("Y")


ax = fig.add_subplot(2, 2, 3)
ax.contourf(xx2, zz2, Z2, alpha=0.4)
ax.scatter(X[:, 0], X[:, 2], c=y, s=20, edgecolor='k')
ax.set_xlabel("X")
ax.set_ylabel("Z")

ax = fig.add_subplot(2, 2, 4)
ax.contourf(yy3, zz3, Z3, alpha=0.4)
ax.scatter(X[:, 1], X[:, 2], c=y, s=20, edgecolor='k')
ax.set_xlabel("Y")
ax.set_ylabel("Z")

"""
plt.contourf(xx1, yy1, Z1, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
"""


"""
#colonna 1
if len(Z1[:,[0]]) < len(Z2[:,[0]]):
    c1 = Z2[:,[0]].copy()
    c1[:len(Z1[:,[0]])] += Z1[:,[0]]
    c1=c1/2
else:
    c1 = Z1[:,[0]].copy()
    c1[:len(Z2[:,[0]])] += Z2[:,[0]]
    c1=c1/2

#colonna 2
if len(Z1[:,[1]]) < len(Z3[:,[0]]):
    c2 = Z3[:,[0]].copy()
    c2[:len(Z1[:,[1]])] += Z1[:,[1]]
    c2=c2/2
else:
    c2 = Z1[:,[1]].copy()
    c2[:len(Z3[:,[0]])] += Z3[:,[0]]
    c2=c2/2

#colonna 3
if len(Z2[:,[1]]) < len(Z3[:,[1]]):
    c3 = Z3[:,[1]].copy()
    c3[:len(Z2[:,[1]])] += Z2[:,[1]]
    c3=c3/2
else:
    c3 = Z2[:,[1]].copy()
    c3[:len(Z3[:,[1]])] += Z3[:,[1]]
    c3=c3/2

    Z=[c1, c2, c3]
 #   print(Z)

"""

"""
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
"""


#plt.plot(Z)


"""
for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [clf1,clf2],
                        ['Decision Tree (depth=4)', 'Decision Tree (depth=4)',
                         'Kernel SVM', 'Soft Voting']):

    Z1 = clf.predict(np.c_[xx1.ravel(), yy1.ravel()])
    Z2 = clf.predict(np.c_[xx2.ravel(), yy2.ravel()])
    Z3 = clf.predict(np.c_[xx3.ravel(), yy3.ravel()])
        
    Z1 = Z1.reshape(xx1.shape)
    Z2 = Z2.reshape(xx2.shape)
    Z3 = Z3.reshape(xx3.shape)
    
    print(Z2)

    Z = (Z1+Z2+Z3)/3

    axarr[idx[0], idx[1]].contourf(xx, yy, zz, Z, alpha=0.4)
    axarr[idx[0], idx[1],idx[2]].scatter(X[:, 0], X[:, 1], Z[:,2], c=y,
                                  s=20, edgecolor='k')
    #axarr[idx[0], idx[1],id].set_title(tt)
"""
plt.show()