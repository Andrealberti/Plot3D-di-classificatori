# Libreria di funzioni Python per il plot 3D di classificatori

Indice delle funzioni:

•	plotDecision2D();

•	plotDecision3D();

•	plotDecision3DConvexHull();

-plotDecision2D()

Input 

•	Training Set -> dati sul quale addestro il classificatore, essi dovranno essere divisi all'interno in dat e target

•	clf -> Rappresenta il classificatore che vogliamo usare

•	DDD -> Valore booleano che ci differenzia se in uscita vogliamo solo i grafici 2D, o anche la loro trasposizione nel 3D

Output 

Grafici risultanti dalla classificazione

Descrizione generale

Questa funzione elabora 3 griglie in 2 dimensioni, ognuna delle quali è combinazioni degli assi ossia XY, XZ, YZ. Ne effettua la predizione tramite classificatore addestrato 

sui due specifici assi del training set. Infine ne plotta i risultati dando la possibilità di portare in 3D il grafico.

-plotDecision3D()

Input 

•	Training Set -> dati sul quale addestro il classificatore, essi dovranno essere divisi all'interno in dat e target

•	clf -> Rappresenta il classificatore che vogliamo usare

•	PassoGriglia -> indica il passo che utilizzeremo per costruire la griglia di punti su cui faremo la predizione. ATTENZIONE: MINORE SARA', MAGGIORE SARA’ LO SFORZO 

COMPUTAZIONALE DEL CALCOLATORE (consigliato [0,1;0,4])

Output 

Grafici 3D risultanti dalla classificazione in cui è compresa la visualizzazione regione per regione.

Descrizione generale

Costruiamo la funzione per plottare le regioni in 3 dimensioni seguendo il seguente schema. Creo una griglia cubica che racchiuda tutti i punti del training set, applico la 

predizione su questa griglia, la divido in N regioni, dove N rappresenta il numero di etichette dato dal training set. Una volta fatto questo vado a plottare le regioni punto 

per punto.

-plotDecision3DConvexHull()

Input 

•	Training Set -> dati sul quale addestro il classificatore, essi dovranno essere divisi all'interno in dat e target

•	clf -> Rappresenta il classificatore che vogliamo usare

•	PassoGriglia -> indica il passo che utilizzeremo per costruire la griglia di punti su cui faremo la predizione. ATTENZIONE: MINORE SARA', MAGGIORE SARA’ LO SFORZO 

COMPUTAZIONALE DEL CALCOLATORE (consigliato [0,1;0,4])

Output 

Grafici 3D risultanti dalla classificazione in cui è compresa la visualizzazione regione per regione.

Descrizione generale

Costruiamo la funzione per plottare le regioni in 3 dimensioni seguendo il seguente schema. Creo una griglia cubica che racchiuda tutti i punti del training set, applico la 

predizione su questa griglia, la divido in N regioni, dove N rappresenta il numero di etichette dato dal training set. Una volta fatto ciò, effettuo un’ approssimazione del 

poligono che li racchiude e plotto il risultato.

