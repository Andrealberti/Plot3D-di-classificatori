import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from mlxtend.data import iris_data
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
#from decision_regions3d import plot_decision_regions

iris, y = iris_data()
"""
#iris = datasets.load_iris().data
#print(iris)

#plt.figure(2, figsize=(10,10))
#pulisco la figura di partenza

plt.clf()

#plt.scatter(iris[:, 0], iris[:, 1])
#a questo punto ho il mio grafico minimale su 2 dimensioni
"""
plt.clf()
clf1 = LogisticRegression(random_state=0)

X = iris[:,[0, 3]]
clf1.fit(X,y)





fig = plt.figure(1, figsize=(10, 10))
ax = fig.add_subplot(2, 2, 1)
#fig = plot_decision_regions(X=X, y=y, clf=clf1, legend=2)
fig = plot_decision_regions(X=X, y=y, clf=clf1, legend=2)

#ax.set_title("logisticRegression")


fig = plt.figure(1, figsize=(10, 10))

#applico la PCA ai miei dati
irisPCA = PCA(n_components=3).fit_transform(iris.data)

#ax = Axes3D(fig, elev=-150, azim=110)
ax = fig.add_subplot(2, 2, 2, projection='3d')

#faccio subplot senza pca
ax.scatter(iris[:, 0], iris[:, 1], iris[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title("Without PCA")

ax = fig.add_subplot(2, 2, 3, projection='3d')

ax.scatter(irisPCA[:, 0], irisPCA[:, 1], irisPCA[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title("With PCA")
"""
"""
clf1 = LogisticRegression(random_state=0)

X = iris[:,[0, 2]]
clf1.fit(X,y)

"""
ax = fig.add_subplot(2, 2, 4, projection='3d')


fig = plt.figure(1, figsize=(8, 6))
fig = plot_decision_regions(X=X, y=y,
                                clf=clf1, legend=2)

ax.set_title("logisticRegression")
"""

plt.show()