from itertools import product
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from sklearn.decomposition import PCA

from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull



def plotDecision2D(dataSet,clf,DDD):

    #divido il trainingSet
    X = dataSet.data
    y = dataSet.target

    #applico la PCA per portarlo a 3 dimensioni
    X = PCA(n_components=3).fit_transform(X)


    #creo le regioni che vorrò valutare, prendo ogni punto disponibile in una grigia di 2 o 3 dimensioni
    #con passo di 0,1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    xx1, yy1 = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))

    xx2, zz2= np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(z_min, z_max, 0.1))

    yy3, zz3= np.meshgrid(np.arange(y_min, y_max, 0.1),
                        np.arange(z_min, z_max, 0.1))


    #addestro il classificatore in 2d ogni volta con gli assi che mi interessano
    clf.fit(X[:,[0,1]], y)
    Z1 = clf.predict(np.c_[xx1.ravel(), yy1.ravel()])

    clf.fit(X[:,[0,2]], y)
    Z2 = clf.predict(np.c_[xx2.ravel(), zz2.ravel()])

    clf.fit(X[:,[1,2]], y)
    Z3 = clf.predict(np.c_[yy3.ravel(), zz3.ravel()])

    #Ho ottenuto le 3 possibili combinazioni delle regioni di decisione sugli assi in 2d
    Z1 = Z1.reshape(xx1.shape)   
    Z2 = Z2.reshape(xx2.shape)
    Z3 = Z3.reshape(yy3.shape)

    #righe di codice necessarie per la contourf

    if DDD==True:  

        fig = plt.figure(1, figsize=(12, 12))

    #grafico xy 3d
        ax = fig.add_subplot(3, 2, 1, projection='3d')
        cset = ax.contourf(xx1, yy1, Z1, levels=np.arange(Z1.max() + 2) - 0.5, alpha=0.4)
        ax.contour(xx1, yy1, Z1, cset.levels, linewidths=0.5, edgecolor='k', antialiased=True )
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

        ax = fig.add_subplot(3, 2, 2)
        cset = ax.contourf(xx1, yy1, Z1, levels=np.arange(Z1.max() + 2) - 0.5, alpha=0.4)
        ax.contour(xx1, yy1, Z1, cset.levels, linewidths=0.5, edgecolor='k', antialiased=True )
        ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    #grafico xz 3d
        ax = fig.add_subplot(3, 2, 3, projection='3d')
        cset = ax.contourf(xx2, zz2, Z2, levels=np.arange(Z2.max() + 2) - 0.5, alpha=0.4)
        ax.contour(xx2, zz2, Z2, cset.levels, linewidths=0.5, edgecolor='k', antialiased=True )
        ax.scatter(X[:, 0], X[:, 2], X[:, 1], c=y, s=20, edgecolor='k')
        ax.view_init(105, -90)
        ax.set_title("XZ")
        ax.set_xlabel("X")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("Z")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("Y")
        ax.w_zaxis.set_ticklabels([])

    #grafico xz 2d
        ax = fig.add_subplot(3, 2, 4)
        cset = ax.contourf(xx2, zz2, Z2, levels=np.arange(Z2.max() + 2) - 0.5, alpha=0.4)
        ax.contour(xx2, zz2, Z2, cset.levels, linewidths=0.5, edgecolor='k', antialiased=True )
        ax.scatter(X[:, 0], X[:, 2], c=y, s=20, edgecolor='k')
        ax.set_xlabel("X")
        ax.set_ylabel("Z")

    #grafico yz 3d
        ax = fig.add_subplot(3, 2, 5, projection='3d')
        cset = ax.contourf(yy3, zz3, Z3, levels=np.arange(Z3.max() + 2) - 0.5, alpha=0.4)
        ax.contour(yy3, zz3, Z3, cset.levels, linewidths=0.5, edgecolor='k', antialiased=True )
        ax.scatter(X[:, 1], X[:, 2], X[:, 0], c=y, s=20, edgecolor='k')
        ax.view_init(105, -90)
        ax.set_title("YZ")
        ax.set_xlabel("Y")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("Z")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("X")
        ax.w_zaxis.set_ticklabels([])

    #grafico yz 2d
        ax = fig.add_subplot(3, 2, 6)
        cset = ax.contourf(yy3, zz3, Z3, levels=np.arange(Z3.max() + 2) - 0.5, alpha=0.4)
        ax.contour(yy3, zz3, Z3, cset.levels, linewidths=0.5, edgecolor='k', antialiased=True )
        ax.scatter(X[:, 1], X[:, 2], c=y, s=20, edgecolor='k')
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
    else :

        fig = plt.figure(1,figsize=(8, 6))
        
    #grafico 2d di xy
        ax = fig.add_subplot(2, 2, 1)
        cset = ax.contourf(xx1, yy1, Z1, levels=np.arange(Z1.max() + 2) - 0.5, alpha=0.4)
        ax.contour(xx1, yy1, Z1, cset.levels, linewidths=0.5, edgecolor='k', antialiased=True)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    #grafico xz 2d
        ax = fig.add_subplot(2, 2, 2)
        cset = ax.contourf(xx2, zz2, Z2, levels=np.arange(Z2.max() + 2) - 0.5, alpha=0.4)
        ax.contour(xx2, zz2, Z2, cset.levels, linewidths=0.5, edgecolor='k', antialiased=True )
        ax.scatter(X[:, 0], X[:, 2], c=y, s=20, edgecolor='k')
        ax.set_xlabel("X")
        ax.set_ylabel("Z")

    #grafico yz 2d
        ax = fig.add_subplot(2, 2, 3)
        cset = ax.contourf(yy3, zz3, Z3, levels=np.arange(Z3.max() + 2) - 0.5, alpha=0.4)
        ax.contour(yy3, zz3, Z3, cset.levels, linewidths=0.5, edgecolor='k', antialiased=True )
        ax.scatter(X[:, 1], X[:, 2], c=y, s=20, edgecolor='k')
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")

    plt.show()






def PlotConvexHull3D(TrainingSet,clf):
    # Loading some example data

    X = TrainingSet.data
    y = TrainingSet.target

    X = PCA(n_components=3).fit_transform(X)


    # Plotting decision regions
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1


    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.5),
                            np.arange(y_min, y_max, 0.5),
                            np.arange(z_min, z_max, 0.5))


    clf.fit(X, y)
    Z=clf.predict(np.c_[xx.ravel(), yy.ravel(),zz.ravel()])

    #utilizzo il ravel per mettere su una sola dimensione i relativi assi, voglio creare per una matrice di 
    #4 colonne nel quale le prime 3 indicano xyz e l'ultima indichi il risultato della predizione

    x1=xx.ravel()
    y1=yy.ravel()
    z1=zz.ravel()
    zz1=Z.ravel()

    Results=np.zeros((len(x1),4))
    Results[:,0]=x1
    Results[:,1]=y1
    Results[:,2]=z1
    Results[:,3]=zz1

    #mi serve sapere quanti sono i possibili risultati del classificatore cosi da costruire in maniera
    #adeguata la lista di plot
    Ylist=y.tolist()
    ValoriY = [x for i, x in enumerate(Ylist) if i == Ylist.index(x)]

    RList = []
    RegionList = []
    HullList = []

    for i in range(len(ValoriY)):
        RList.append([(t[0],t[1],t[2]) for t in Results if t[3]==ValoriY[i]])
        RegionList.append(np.array(RList[-1]))    #-1 prende l'ultimo elemento della lista
        HullList.append(ConvexHull(RegionList[-1]))



    fig = plt.figure()
    for i in range(len(ValoriY)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.plot_trisurf(RegionList[i][:,0],RegionList[i][:,1],RegionList[i][:,2], triangles=HullList[i].simplices, alpha=0.4,color='purple')
        ax.scatter(RegionList[i][:,0],RegionList[i][:,1],RegionList[i][:,2], alpha=0.4, marker="s")
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        ax.set_zlim(z_min,z_max)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    for i in range(len(ValoriY)):
        ax.plot_trisurf(RegionList[i][:,0],RegionList[i][:,1],RegionList[i][:,2], triangles=HullList[i].simplices, alpha=0.4)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=20, edgecolor='k')
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_zlim(z_min,z_max)

    plt.show()