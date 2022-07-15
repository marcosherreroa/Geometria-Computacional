# -*- coding: utf-8 -*-
"""
Práctica 3 de Geometría Computacional

Autor: Marcos Herrero
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans,DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from scipy.spatial import Voronoi, voronoi_plot_2d


"""
Funciones auxiliares
"""

#Representa los elementos distribuidos en clusters y, opcionalmente,
#el diagrama de Voronoi asociado
def representarClusters(title, filename, labels, centers = None):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    plt.figure(figsize=(8,4))

    for k, col in zip(unique_labels, colors):
        
        size = 4
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
            size = 2

        class_member_mask = (labels == k)

        members = X[class_member_mask]
        plt.plot(members[:, 0], members[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=size)

    axes = plt.gca()
    
    if not(centers is None):
        vor = Voronoi(centers)
        fig = voronoi_plot_2d(vor,ax=axes)

    plt.title(title)
    axes.set_xlim(-2.25,2.5)
    axes.set_ylim(-2.5,1.75)
    fig = plt.gcf()
    fig.savefig(filename,format='pdf')
    plt.show()  

#Representa los clusters obtenidos por KMeans
def representarClustersKMeans(kmeans,filename):
    title = "Clasificación KMeans con k = {}".format(kmeans.n_clusters)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    representarClusters(title, filename, labels, centers)

#Representa los clusters obtenidos por DBSCAN
def representarClustersDBSCAN(db,filename):
    title =''
    
    if db.metric == 'euclidean':
        title = "Clasificación DBSCAN con métrica euclídea y eps = {}".format(
            db.eps)
    elif db.metric == 'manhattan':
        title = "Clasificación DBSCAN con métrica Manhattan y eps = {}".format(
            db.eps)
    
    labels = db.labels_
    representarClusters(title, filename, labels)

#Función que clasifica el sistema X usando KMeans para diferentes valores de k
#y devuelve la clasificación que da un mayor coeficiente de silhouette. 
#También muestra gráficamente el coeficiente de silhouette obtenido para
# cada valor de k
def resolverKMeans(X):
    optimkmeans = None
    optimsilhouette = -2
    silhouette = np.empty(len(range(2,16)))

    for k in range(2,16):
        
        kmeans = KMeans(n_clusters= k,random_state=0).fit(X)
        labels = kmeans.labels_
        silhouette[k-2] = metrics.silhouette_score(X, labels)
        
        if silhouette[k-2] > optimsilhouette:
            optimsilhouette = silhouette[k-2]
            optimkmeans = kmeans
    
    print("Número de clusters óptimo: {}".format(optimkmeans.n_clusters))
    print("Coeficiente de Silhouette: {}".format(optimsilhouette))

    plt.title("Coeficiente de Silhouette para KMeans")
    plt.plot(range(2,16),silhouette)
    plt.xlabel("Núm. clusters")
    plt.ylabel("Coef. de Silhouette")
    fig = plt.gcf()
    fig.savefig("silhouettekmeans.pdf",format='pdf')
    plt.show()
    
    
    
    return optimkmeans

#Función que resuelve el apartado ii) para el sistema X
def resolverDBSCAN(X,metrica):
    optimdb = None
    optimsilhouette = -2
    silhouette = np.empty(10)
    ind = 0

    for epsilon in np.linspace(0.1,0.4,10):
        
        db = DBSCAN(eps=epsilon, min_samples=10, metric=metrica).fit(X)
        labels = db.labels_
        silhouette[ind] = metrics.silhouette_score(X, labels)
        
        if silhouette[ind] > optimsilhouette:
            optimsilhouette = silhouette[ind]
            optimdb = db
        
        ind += 1
    
    print("Epsilon óptimo: {}".format(optimdb.eps))
    print("Coeficiente de Silhouette: {}".format(optimsilhouette))

    title =''
    filename=''
    if metrica == 'euclidean':
        title = "Coeficiente de Silhouette para DBSCAN con métrica euclídea"
        filename = "silhouettedbscaneucl.pdf"
        
    elif metrica == 'manhattan':
        title = "Coeficiente de Silhouette para DBSCAN con métrica Manhattan"
        filename = "silhouettedbscanmanh.pdf"
        
    plt.title(title)
    plt.plot(np.linspace(0.1,0.4,10),silhouette)
    plt.xlabel("Núm. clusters")
    plt.ylabel("Coef. de Silhouette")
    fig = plt.gcf()
    fig.savefig(filename,format='pdf')
    plt.show()
    
    return optimdb

""""
Generar el sistema
"""
centers = [[-0.5, 0.5], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,
                            random_state=0)

"""
Apartado i): Obtener el número de clusters óptimo utilizando KMeans y el 
coeficiente de Silhouette . Mostrar en un gráfica el coeficiente de Silhouette
para k =2,3,...,15. Representar la clasificación para el número óptimo de 
clusters y el  diagrama de Voronoi en la misma gráfica.
"""

print("Apartado i):")
kmeans = resolverKMeans(X)     
representarClustersKMeans(kmeans, "kmeans.pdf")
print()

"""
Apartado ii): Obtener el umbral de distancia epsilon óptimo utilizando el
DBSCAN y el coeficiente de Silhouette. Hacerlo con métrica euclidean y luego
con manhattan y comparar con la gráfica del apartado anterior
"""

print("Apartado ii):")

#Con métrica euclídea
print("Con métrica euclídea:")
dbeuclid = resolverDBSCAN(X,'euclidean')
representarClustersDBSCAN(dbeuclid,"dbscaneuclid.pdf")
print()

#Con métrica Manhattan
print("Con métrica Manhattan:")
dbmanh = resolverDBSCAN(X,'manhattan')
representarClustersDBSCAN(dbmanh,"dbscanmanh.pdf")
print()

"""
Apartado iii): Decidir a qué vecindad pertenecen los puntos a = (0,0) y
b = (0,-1)   
"""

print("Apartado iii):")

a= [0,0]
b = [0,-1]

#Con kmeans
print("Según KMeans:")

clustera = kmeans.predict([a])
centroclusta = kmeans.cluster_centers_[clustera]
print("* El elemento a = {} pertenece al cluster {}, de centro {}".format(a,
        clustera[0],centroclusta))

clusterb = kmeans.predict([b])
centroclustb = kmeans.cluster_centers_[clusterb]
print("* El elemento b = {} pertenece al cluster {}, de centro {}".format(b,
        clusterb[0],centroclustb))