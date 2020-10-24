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
from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull
import mpl_toolkits.mplot3d as a3



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


xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1),
                         np.arange(z_min, z_max, 0.1))

clf.fit(X, y)
Z=clf.predict(np.c_[xx.ravel(), yy.ravel(),zz.ravel()])

Z=Z.reshape(xx.shape)


x1=xx.ravel()
y1=yy.ravel()
z1=zz.ravel()
zz1=Z.ravel()


Results=np.zeros((len(x1),4))
print("Results shape",Results.shape)
Results[:,0]=x1
Results[:,1]=y1
Results[:,2]=z1
Results[:,3]=zz1


print("")
r0= [(t[0],t[1],t[2]) for t in Results if t[3]==0]
Region0 = np.array(r0)
hull_Region0 = ConvexHull(Region0)
hull_facets_Region0 = hull_Region0.points[hull_Region0.simplices]

r1= [(t[0],t[1],t[2]) for t in Results if t[3]==1]
Region1 = np.array(r1)
hull_Region1 = ConvexHull(Region1)
hull_facets_Region1 = hull_Region1.points[hull_Region1.simplices]

r2= [(t[0],t[1],t[2]) for t in Results if t[3]==2]
Region2 = np.array(r2)
hull_Region2 = ConvexHull(Region2)
hull_facets_Region2 = hull_Region2.points[hull_Region2.simplices]

print(hull_Region2.simplices)



fig = plt.figure()
"""
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_trisurf(Region0[:,0], Region0[:,1], Region0[:,2], triangles=hull_Region0.simplices, alpha=0.4,color='purple')
ax.scatter(Region0[:,0],Region0[:,1],Region0[:,2], alpha=0.4)
ax.set_xlim(x_min,x_max)
ax.set_ylim(y_min,y_max)
ax.set_zlim(z_min,z_max)

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot_trisurf(Region1[:,0], Region1[:,1], Region1[:,2], triangles=hull_Region1.simplices, alpha=0.4,color='blue')
ax.scatter(Region1[:,0],Region1[:,1],Region1[:,2], alpha=0.4)
ax.set_xlim(x_min,x_max)
ax.set_ylim(y_min,y_max)
ax.set_zlim(z_min,z_max)

ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.plot_trisurf(Region2[:,0], Region2[:,1], Region2[:,2], triangles=hull_Region2.simplices, alpha=0.4, color='yellow')
ax.scatter(Region2[:,0],Region2[:,1],Region2[:,2], alpha=0.4)
ax.set_xlim(x_min,x_max)
ax.set_ylim(y_min,y_max)
ax.set_zlim(z_min,z_max)
"""
ax = fig.add_subplot(1,1,1, projection='3d')

ax.plot_trisurf(Region0[:,0], Region0[:,1], Region0[:,2], triangles=hull_Region0.simplices, alpha=0.3,color='purple')
ax.plot_trisurf(Region1[:,0], Region1[:,1], Region1[:,2], triangles=hull_Region1.simplices, alpha=0.3,color='blue')
ax.plot_trisurf(Region2[:,0], Region2[:,1], Region2[:,2], triangles=hull_Region2.simplices, alpha=0.3, color='yellow')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=20, edgecolor='k')

#ax.scatter(Results[:,0],Results[:,1],Results[:,2], alpha=0.4, c=Results[:,3], s=20, edgecolor='k')
ax.set_xlim(x_min,x_max)
ax.set_ylim(y_min,y_max)
ax.set_zlim(z_min,z_max)

plt.show()
"""
fig = plt.figure(1, figsize=(12, 12))
ax = fig.add_subplot(1, 2, 1, projection='3d')

ax.scatter(xx[:,:,:],yy[:,:,:],zz[:,:,:], alpha=0.4, c=Z[:,:,:], s=20, edgecolor='k')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(x1,y1,z1, alpha=0.4, c=zz1, s=20, edgecolor='k')

#ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=20, edgecolor='k')
plt.show()
"""

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