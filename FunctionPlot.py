import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

#plotDecision2D
#Questa funzione elabora 3 griglie in 2 dimensioni, ognuna delle quali è combinazioni degli assi ossia
# XY, XZ, YZ. Ne effettua la predizione tramite classificatore addestrato sui due specifici assi del
#training set. Infine ne plotta i risultati dando la possibilità di portare in 3D il grafico 

#input funzione
#Training Set -> dati sul quale addestro il classificatore, essi dovranno essere divisi all'interno in data
#                e target
# clf -> Rappresenta il classificatore che vogliamo usare
# DDD -> Valore booleano che ci differenzia se in uscita vogliamo solo i grafici 2D, o anche la loro trasposizione
#        nel 3D

def plotDecision2D(dataSet,clf,DDD):

    #inizio dividendo il mio training set in una matrice contenente le posizioni dei punti ed un contenente
    #la loro etichetta
    X = dataSet.data
    y = dataSet.target

    #applico la PCA per portarlo a 3 dimensioni
    X = PCA(n_components=3).fit_transform(X)

    #creo le regioni che vorrò valutare, prendo ogni punto disponibile in una grigia di 2 o 3 dimensioni
    #con passo di 0,1, l'idea è creare un cubo che coprà in quale volume racchiuda tutto il mio training set
    #e valutare passo per passo ogni punto appartenente a questo cubo
    #inizio prendendo gli 'estremi' del cubo
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    #costruisco 3 grigle di dimensione 2, voglio valutare il classificatore su xy, xz, yz
    xx1, yy1= np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    xx2, zz2= np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(z_min, z_max, 0.1))
    yy3, zz3= np.meshgrid(np.arange(y_min, y_max, 0.1), np.arange(z_min, z_max, 0.1))


    #addestro il classificatore in 2d ogni volta con gli assi che mi interessano
    #voglio ora valutare e predirre i punti appartente a queste griglie create. ogni volta addesto il classificatore
    #sugli assi del training set che sto valutando
    clf.fit(X[:,[0,1]], y)
    Z1 = clf.predict(np.c_[xx1.ravel(), yy1.ravel()])

    clf.fit(X[:,[0,2]], y)
    Z2 = clf.predict(np.c_[xx2.ravel(), zz2.ravel()])

    clf.fit(X[:,[1,2]], y)
    Z3 = clf.predict(np.c_[yy3.ravel(), zz3.ravel()])

    #Ho ottenuto le 3 possibili combinazioni delle regioni di decisione sugli assi in 2d, le riporto in dimensione
    #facilmente utilizzabile nel mio grafico ossia vettore a una dimensione
    Z1 = Z1.reshape(xx1.shape)   
    Z2 = Z2.reshape(xx2.shape)
    Z3 = Z3.reshape(yy3.shape)

    #questo if decide se plottare oltre al grafico 2d del risultato la veduta 3d di esso, non essendo rilevante
    #lo abbiamo inserito nel caso si volesse allegerire il plot
    #per plottare i nostri risultati utilizziamo la funzione contour
    #utilizzando la falsa riga del codice della libreria MLEXTEND utiliziamo contourf per mostrare a schermo
    #ogni regione come un unica area a se, ben divisa dalle altre
    #ed infine plottiamo i punti del training set all'interno di esse per valutarne il classificatore 
    if DDD==True:  

        fig = plt.figure(1, figsize=(12, 12))

    #grafico xy 3d
        ax = fig.add_subplot(3, 2, 1, projection='3d')
        ax.contourf(xx1, yy1, Z1, levels=np.arange(Z1.max() + 2) - 0.5, alpha=0.4, linewidths=0.5, edgecolor='k', antialiased=True )
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
        ax.contourf(xx1, yy1, Z1, levels=np.arange(Z1.max() + 2) - 0.5, alpha=0.4, linewidths=0.5, edgecolor='k', antialiased=True )
        ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    #grafico xz 3d
        ax = fig.add_subplot(3, 2, 3, projection='3d')
        ax.contourf(xx2, zz2, Z2, levels=np.arange(Z2.max() + 2) - 0.5, alpha=0.4, linewidths=0.5, edgecolor='k', antialiased=True )
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
        ax.contourf(xx2, zz2, Z2, levels=np.arange(Z2.max() + 2) - 0.5, alpha=0.4, linewidths=0.5, edgecolor='k', antialiased=True )
        ax.scatter(X[:, 0], X[:, 2], c=y, s=20, edgecolor='k')
        ax.set_xlabel("X")
        ax.set_ylabel("Z")

    #grafico yz 3d
        ax = fig.add_subplot(3, 2, 5, projection='3d')
        ax.contourf(yy3, zz3, Z3, levels=np.arange(Z3.max() + 2) - 0.5, alpha=0.4, linewidths=0.5, edgecolor='k', antialiased=True )
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
        ax.contourf(yy3, zz3, Z3, levels=np.arange(Z3.max() + 2) - 0.5, alpha=0.4, linewidths=0.5, edgecolor='k', antialiased=True )
        ax.scatter(X[:, 1], X[:, 2], c=y, s=20, edgecolor='k')
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
    else :

        fig = plt.figure(1,figsize=(8, 6))
        
    #grafico 2d di xy
        ax = fig.add_subplot(2, 2, 1)
        ax.contourf(xx1, yy1, Z1, levels=np.arange(Z1.max() + 2) - 0.5, alpha=0.4, linewidths=0.5, edgecolor='k', antialiased=True)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    #grafico xz 2d
        ax = fig.add_subplot(2, 2, 2)
        ax.contourf(xx2, zz2, Z2,  levels=np.arange(Z2.max() + 2) - 0.5, alpha=0.4, linewidths=0.5, edgecolor='k', antialiased=True )
        ax.scatter(X[:, 0], X[:, 2], c=y, s=20, edgecolor='k')
        ax.set_xlabel("X")
        ax.set_ylabel("Z")

    #grafico yz 2d
        ax = fig.add_subplot(2, 2, 3)
        ax.contourf(yy3, zz3, Z3, levels=np.arange(Z3.max() + 2) - 0.5, alpha=0.4, linewidths=0.5, edgecolor='k', antialiased=True )
        ax.scatter(X[:, 1], X[:, 2], c=y, s=20, edgecolor='k')
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")

    plt.show()


#PlotConvexHull3D
#Costruimo la funzione per plottare le regioni in 3 dimensioni seguendo il seguente schema
#Creo una griglia cubica che racchiuda tutti i punti del training set, applico la predizione su questa griglia,
#la divido in N regioni, dove N rappresenta il numero di etichette dato dal training set. Una volta fatto questo
#vado a plottare le regioni punto per punto

#input funzione
#Training Set -> dati sul quale addestro il classificatore, essi dovranno essere divisi all'interno in data
#                e target
# clf -> Rappresenta il classificatore che vogliamo usare
# PassoGriglia -> indica il passo che utilizzeremo per costruire la griglia di punti su cui faremo la predizione                 
#                 ATTENZIONE: MINORE SARA', MAGGIORE SARà LO SFORZO COMPUTAZIONALE DEL CALCOLATORE 
#                 consigliato [0,1 0,4]

def PlotConvexHull3D(TrainingSet,clf,PassoGriglia):

    #inizio dividendo il mio training set in una matrice contenente le posizioni dei punti ed un contenente
    #la loro etichetta
    X = TrainingSet.data
    y = TrainingSet.target

    #applico la PCA per riportare i dati del training set in 3 dimensioni
    X = PCA(n_components=3).fit_transform(X)

    #creo le regioni che vorrò valutare, prendo ogni punto disponibile di una griglia 3 dimensioni
    #con passo da impostare (esempio 0.1). L'idea è creare un cubo il quale volume racchiuda
    # tutto il mio training set e valutare passo per passo ogni punto appartenente a questo cubo
    #inizio prendendo gli 'estremi' del cubo
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    #creo la mia griglia a 3 dimensioni che andrò successivamente a valuterò con il classificatore
    #il parametro PassoGriglia indica il passo con cui devo prendere il punto successivo nella griglia
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, PassoGriglia),
                            np.arange(y_min, y_max, PassoGriglia),
                            np.arange(z_min, z_max, PassoGriglia))

    #addestro il classificatore e valuto il mio cubo
    clf.fit(X, y)
    Z=clf.predict(np.c_[xx.ravel(), yy.ravel(),zz.ravel()])

    #utilizzo il ravel per mettere su una sola dimensione i relativi assi, voglio creare per una matrice di 
    #4 colonne nel quale le prime 3 indicano xyz e l'ultima indichi il risultato della predizione
    x1=xx.ravel()
    y1=yy.ravel()
    z1=zz.ravel()
    zz1=Z.ravel()

    #definisco la matrice e successivamente la riempo
    Results=np.zeros((len(x1),4))
    Results[:,0]=x1
    Results[:,1]=y1
    Results[:,2]=z1
    Results[:,3]=zz1

    #mi serve sapere quanti sono i possibili risultati del classificatore cosi da costruire in maniera
    #adeguata la lista di plot
    Ylist=y.tolist()
    ValoriY = [x for i, x in enumerate(Ylist) if i == Ylist.index(x)]

    #definisco 2 liste vuote che andrò successivente a riempire con le regioni
    RegionList = []

    #ora che ho la lista di quali sono le possibili regioni, vado a riempirle ricavandole dalla matrice 
    #result creata 2 passi prima.
    #Ciclo per ogni possibile valore trovato nelle etichette. Per ogni valore ricavo dalla matrice Result
    #le prime 3 colonne che corrispondono alle coordinate XYZ. Ora vado ad aggiungere alla lista RegionList
    #i valoti appena ricavati, racchiusi in un array contente i suddetti punti.
    #Infine effettuo la convexHull su questi valori cosi da ottenere un volume approssimato contenente i punti
    #trovati per ogni regione.
    for i in range(len(ValoriY)):
        RegionList.append(np.array([(t[0],t[1],t[2]) for t in Results if t[3]==ValoriY[i]]))   
        

    #applico lo stesso ragionamento ai dati del training set cosi da raprresentarli nelle giuste regioni
    #li raggruppo in un array a 4 colonne e poi estraggo le coordinate di ogni punto regione per regione
    TrainingRegion = []
    TResult = np.c_[X, y]
    for i in range(len(ValoriY)):
        TrainingRegion.append(np.array([(t[0],t[1],t[2]) for t in TResult if t[3]==ValoriY[i]])) 

    #ora che ho fatto le mie previsione provvedo ai plot
    #effettuo 4 figure
    #1) mostriamo tramite i punti del cubo le varie regioni, ognuna ha il suo grafico con anche i punti
    #   relativi del training set appartenenti ad essa
    #2) raggruppiamo tutte le regioni nello steso grafico
    #3) Ripeto il ragionamento precedente ma questa volta utilizzando la tecnica calcolata prima della ConvexHull
    #   Dobbiammo fare attenzione perchè anche se il grafico ottenuto è più limpido nella lettura si porta dietro
    #   il concetto di convex hull ossia la nostra regione viene approssimata nel volume rappresentato
    #   ESSO E' APPROSSIMATO QUINDI NON PRECISO. INFATTI NEL GRAFICO 4 SI POTRANNO VEDERE SOVRAPPOSIZIONI   
    #4) Raggruppo le rappresentazioni di convexHull in un unico grafico

    #seleziono la mappa dei coloti per i plot
    jet= plt.get_cmap('jet')




    fig = plt.figure(1)
    #seleziono tanti colori quante sono le regioni. 
    colors = iter(jet(np.linspace(0,(150/len(ValoriY)),150)))
    
    #divido il plot in 2, creo Lplot per avere il numero giusto di subplot
    Lplot= int(len(ValoriY)/2) + (len(ValoriY) % 2 > 0)   #ho bisogno del valore/2 sempre preso per eccesso
    
    #per ogni regione rappresento i suoi punti appartententi al cubo e quelli del training set
    for i in range(len(ValoriY)):
        ax = fig.add_subplot(Lplot, 2, i+1, projection='3d')
        ax.scatter(RegionList[i][:,0],RegionList[i][:,1],RegionList[i][:,2], alpha=0.015,
                                                           marker="s",color=next(colors))
        ax.scatter(TrainingRegion[i][:,0],TrainingRegion[i][:,1],TrainingRegion[i][:,2], color="black")
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        ax.set_zlim(z_min,z_max)
        plt.title("Region %s" % str(i+1))


    fig = plt.figure(2)
    #seleziono tanti colori quante sono le regioni. 
    colors = iter(jet(np.linspace(0,(150/len(ValoriY)),150)))
    
    #divido il plot in 2, creo Lplot per avere il numero giusto di subplot
    Lplot= int(len(ValoriY)/2) + (len(ValoriY) % 2 > 0)   #ho bisogno del valore/2 sempre preso per eccesso
    
    #per ogni regione rappresento i suoi punti appartententi al cubo e quelli del training set
    for i in range(len(ValoriY)):
        ax = fig.add_subplot(Lplot, 2, i+1, projection='3d')
        ax.scatter(RegionList[i][:,0],RegionList[i][:,1],RegionList[i][:,2], alpha=0.015,
                                                           marker="s",color=next(colors))
        ax.scatter(X[:,0],X[:,1],X[:,2],c=y)
                     
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        ax.set_zlim(z_min,z_max)
        plt.title("Region %s" % str(i+1))


    fig = plt.figure(3)
    colors = iter(jet(np.linspace(0,(150/len(ValoriY)),150)))
    ax = fig.add_subplot(1,1,1, projection='3d')
    #raggruppo sotto lo stesso grafico tutte le regioni
    for i in range(len(ValoriY)):
        ax.scatter(RegionList[i][:,0],RegionList[i][:,1],RegionList[i][:,2], alpha=0.015, 
                                                          marker="s",  color=next(colors))
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_zlim(z_min,z_max)
    plt.title("All region")

    plt.show()



def PlotConvexHull3DConvexHull(TrainingSet,clf,PassoGriglia):
    
    #inizio dividendo il mio training set in una matrice contenente le posizioni dei punti ed un contenente
    #la loro etichetta
    X = TrainingSet.data
    y = TrainingSet.target

    #applico la PCA per riportare i dati del training set in 3 dimensioni
    X = PCA(n_components=3).fit_transform(X)


    #creo le regioni che vorrò valutare, prendo ogni punto disponibile di una griglia 3 dimensioni
    #con passo da impostare (esempio 0.1). L'idea è creare un cubo il quale volume racchiuda
    # tutto il mio training set e valutare passo per passo ogni punto appartenente a questo cubo
    #inizio prendendo gli 'estremi' del cubo
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    #creo la mia griglia a 3 dimensioni che andrò successivamente a valuterò con il classificatore
    #il parametro PassoGriglia indica il passo con cui devo prendere il punto successivo nella griglia
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, PassoGriglia),
                            np.arange(y_min, y_max, PassoGriglia),
                            np.arange(z_min, z_max, PassoGriglia))

    #addestro il classificatore e valuto il mio cubo
    clf.fit(X, y)
    Z=clf.predict(np.c_[xx.ravel(), yy.ravel(),zz.ravel()])

    #utilizzo il ravel per mettere su una sola dimensione i relativi assi, voglio creare per una matrice di 
    #4 colonne nel quale le prime 3 indicano xyz e l'ultima indichi il risultato della predizione
    x1=xx.ravel()
    y1=yy.ravel()
    z1=zz.ravel()
    zz1=Z.ravel()

    #definisco la matrice e successivamente la riempo
    Results=np.zeros((len(x1),4))
    Results[:,0]=x1
    Results[:,1]=y1
    Results[:,2]=z1
    Results[:,3]=zz1

    #mi serve sapere quanti sono i possibili risultati del classificatore cosi da costruire in maniera
    #adeguata la lista di plot
    Ylist=y.tolist()
    ValoriY = [x for i, x in enumerate(Ylist) if i == Ylist.index(x)]

    #definisco 2 liste vuote che andrò successivente a riempire con le regioni
    RegionList = []
    HullList = []

    #ora che ho la lista di quali sono le possibili regioni, vado a riempirle ricavandole dalla matrice 
    #result creata 2 passi prima.
    #Ciclo per ogni possibile valore trovato nelle etichette. Per ogni valore ricavo dalla matrice Result
    #le prime 3 colonne che corrispondono alle coordinate XYZ. Ora vado ad aggiungere alla lista RegionList
    #i valoti appena ricavati, racchiusi in un array contente i suddetti punti.
    #Infine effettuo la convexHull su questi valori cosi da ottenere un volume approssimato contenente i punti
    #trovati per ogni regione.
    for i in range(len(ValoriY)):
        RegionList.append(np.array([(t[0],t[1],t[2]) for t in Results if t[3]==ValoriY[i]]))   
        HullList.append(ConvexHull(RegionList[-1])) #-1 prende l'ultimo elemento della lista

    #applico lo stesso ragionamento ai dati del training set cosi da raprresentarli nelle giuste regioni
    #li raggruppo in un array a 4 colonne e poi estraggo le coordinate di ogni punto regione per regione
    TrainingRegion = []
    TResult = np.c_[X, y]
    for i in range(len(ValoriY)):
        TrainingRegion.append(np.array([(t[0],t[1],t[2]) for t in TResult if t[3]==ValoriY[i]])) 

    #ora che ho fatto le mie previsione provvedo ai plot
    #effettuo 4 figure
    #1) mostriamo tramite i punti del cubo le varie regioni, ognuna ha il suo grafico con anche i punti
    #   relativi del training set appartenenti ad essa
    #2) raggruppiamo tutte le regioni nello steso grafico
    #3) Ripeto il ragionamento precedente ma questa volta utilizzando la tecnica calcolata prima della ConvexHull
    #   Dobbiammo fare attenzione perchè anche se il grafico ottenuto è più limpido nella lettura si porta dietro
    #   il concetto di convex hull ossia la nostra regione viene approssimata nel volume rappresentato
    #   ESSO E' APPROSSIMATO QUINDI NON PRECISO. INFATTI NEL GRAFICO 4 SI POTRANNO VEDERE SOVRAPPOSIZIONI   
    #4) Raggruppo le rappresentazioni di convexHull in un unico grafico

    #seleziono la mappa dei coloti per i plot
    jet= plt.get_cmap('jet')


    #seleziono tanti colori quante sono le regioni. 
    colors = iter(jet(np.linspace(0,(150/len(ValoriY)),150)))
    
    #divido il plot in 2, creo Lplot per avere il numero giusto di subplot
    Lplot= int(len(ValoriY)/2) + (len(ValoriY) % 2 > 0)   #ho bisogno del valore/2 sempre preso per eccesso
    

    fig = plt.figure(1)
    colors = iter(jet(np.linspace(0,(150/len(ValoriY)),150)))
    #divido il plot in 2, creo Lplot per avere il numero giusto di subplot
    Lplot= int(len(ValoriY)/2) + (len(ValoriY) % 2 > 0)   #ho bisogno del valore/2 sempre preso per eccesso
    #rappresento tramite triangoli la convexHull calcolata in precedenza, ottengo un volume approssimato
    for i in range(len(ValoriY)):
        ax = fig.add_subplot(Lplot, 2, i+1, projection='3d')
        ax.plot_trisurf(RegionList[i][:,0],RegionList[i][:,1],RegionList[i][:,2],
                         triangles=HullList[i].simplices, alpha=0.4, color=next(colors))
        ax.scatter(TrainingRegion[i][:,0],TrainingRegion[i][:,1],TrainingRegion[i][:,2], color="black")
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        ax.set_zlim(z_min,z_max)
        plt.title("Region %s" % str(i+1))

    fig = plt.figure(2)
    colors = iter(jet(np.linspace(0,(150/len(ValoriY)),150)))
    #divido il plot in 2, creo Lplot per avere il numero giusto di subplot
    Lplot= int(len(ValoriY)/2) + (len(ValoriY) % 2 > 0)   #ho bisogno del valore/2 sempre preso per eccesso
    #rappresento tramite triangoli la convexHull calcolata in precedenza, ottengo un volume approssimato
    for i in range(len(ValoriY)):
        ax = fig.add_subplot(Lplot, 2, i+1, projection='3d')
        ax.plot_trisurf(RegionList[i][:,0],RegionList[i][:,1],RegionList[i][:,2],
                         triangles=HullList[i].simplices, alpha=0.4, color=next(colors))
        ax.scatter(X[:,0],X[:,1],X[:,2],c=y)
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        ax.set_zlim(z_min,z_max)
        plt.title("Region %s" % str(i+1))


    fig = plt.figure(3)
    colors = iter(jet(np.linspace(0,(150/len(ValoriY)),150)))
    ax = fig.add_subplot(1,1,1, projection='3d')
    #rappresento tutti i volumi delle regioni
    for i in range(len(ValoriY)):
        ax.plot_trisurf(RegionList[i][:,0],RegionList[i][:,1],RegionList[i][:,2], 
        triangles=HullList[i].simplices, alpha=0.2,  color=next(colors))
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_zlim(z_min,z_max)
    plt.title("All region (with volume)")

    plt.show()



