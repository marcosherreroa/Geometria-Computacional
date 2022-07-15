# -*- coding: utf-8 -*-
"""
Práctica 4 de Geometría Computacional
Autor: Marcos Herrero
"""

import datetime as dt  # Python standard library datetime  module
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

"""
Cargamos T y Z de 2021 y 2022
"""

#Fichero de T (temperatura) en 2021
f = Dataset("air.2021.nc", "r", format="NETCDF4")

#Rangos : 
# time mapea cada día del año 2021 a horas desde el 01/01/1800
# level mapea cada nivel de presión considerado al número de hPa que representa
# lats mapea cada nivel de latitud al número de grados respecto al ecuador (entre 90 y -90)
# lons mapea cada nivel longitud(meridiano) al número de grados respecto al 
# meridiano de Greenwich (entre 0 y 360)
time21 = f.variables['time'][:].copy()
level = f.variables['level'][:].copy()
lats = f.variables['lat'][:].copy()
lons = f.variables['lon'][:].copy()

#Temperatura en cada posición en Kelvin
air21 = f.variables['air'][:].copy()
f.close()

#Fichero de Z (altura) en 2021
f = Dataset("hgt.2021.nc", "r", format="NETCDF4")
hgt21 = f.variables['hgt'][:].copy()
f.close()

#Fichero de T en  2022 (level,lats y lons son los mismos que en 2021)
f = Dataset("air.2022.nc", "r", format="NETCDF4")
time22 = f.variables['time'][:].copy()

#Temperatura en cada posición en Kelvin
air22 = f.variables['air'][:].copy()
f.close()

#Fichero de Z (altura) en 2022
f = Dataset("hgt.2022.nc", "r", format="NETCDF4")
hgt22 = f.variables['hgt'][:].copy()
f.close()

'''
Vamos a modificar el rango de lons para que España no quede partida.
Necesitamos que lons vaya de -180 a 180, y para ello hemos de reorganizar los 
datos de air21 y hgt21
'''

pos180 = np.argmax(lons >= 180)

air21 = np.roll(air21,axis = 3, shift = -pos180)
hgt21 = np.roll(hgt21,axis = 3, shift = -pos180)
air22 = np.roll(air22,axis = 3, shift = -pos180)
hgt22 = np.roll(hgt22,axis = 3, shift = -pos180)
lons = np.roll(lons, shift= -pos180)
lons[lons >= 180] -= 360

'''
Apartado i): Estimar las 4 componentes principales del sistema fijando 
p = 500 hPa
'''

print('Apartado i):')
print()

n_components = 4

#Nos quedamos con los datos con p = 500 hPa
hgt21b = hgt21[:,level==500.,:,:].reshape(len(time21),len(lats)*len(lons))
air21b = air21[:,level==500.,:,:].reshape(len(time21),len(lats)*len(lons))

#Aplicaremos pca sobre la matriz traspuesta para reducir el número de elementos
# (días) y no el de variables de estado
X = hgt21b.transpose()
pca = PCA(n_components= n_components)


Element_pca = pca.fit_transform(X)
Element_pca = Element_pca.transpose(1,0).reshape(n_components,len(lats),len(lons))

print("Fracción de varianza explicada: {}".format(pca.explained_variance_ratio_))

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    ax.text(0.5, 90, 'PCA-'+str(i),fontsize=18, ha='center')
    plt.contour(lons, lats, Element_pca[i-1,:,:],cmap = 'coolwarm')
    plt.colorbar()
 
fig = plt.gcf()
fig.savefig("contornosi.pdf",format='pdf')   
plt.show()

print()
print()

'''
Apartado ii): consideramos el subsistema con x en (-20,20) e y en (30,50).
Buscar los 4 días de 2021 más análogos de a0 = 2022/01/11 considerando solo Z.
Calcular el error absoluto medio de la  temperatura (con p = 1000 hPa) prevista
para el elemento a0 segun la media de dichos análogos. Para la analogía,
consideramos la distancia euclídea con pesos 1 para x,y , 0,5 para las presiones
de 500 y 1000 hPa y 0 para el resto.
'''

print('Apartado ii)')
print()

#Buscamos el dia a0
a0 = dt.date(2022, 1, 11)
timea0 = (a0 - dt.date(1800,1,1)).total_seconds()/3600
aira0 = air22[time22 == timea0,:,:,:][0]
hgta0 = hgt22[time22 == timea0,:,:,:][0]

#Restringimos el sistema y el día a los rangos pedidos
condlats = np.logical_and(lats > 30,lats < 50)
condlons = np.logical_and(lons > -20, lons < 20)

airsigma = air21[:,:,condlats,:]
airsigma = airsigma[:,:,:,condlons]
hgtsigma = hgt21[:,:,condlats,:]
hgtsigma = hgtsigma[:,:,:,condlons]
aira0rest = aira0[:,condlats,:]
aira0rest = aira0rest[:,:,condlons]
hgta0rest = hgta0[:,condlats,:]
hgta0rest = hgta0rest[:,:,condlons]

latssigma = lats[condlats]
lonssigma = lons[condlons]

#Calculamos los 4 elementos más análogos a a0 en hgt2021
n_neighbours = 4

weights = np.zeros((len(level),len(latssigma),len(lonssigma)))
weights[level == 500,:,:] = 0.5
weights[level == 1000,:,:] = 0.5

neigh = NearestNeighbors(n_neighbors= n_neighbours, metric_params = {'w':weights.flatten()} )
neigh.fit(hgtsigma.reshape(len(time21),len(level)*len(latssigma)*len(lonssigma)))
distsa0, neighboursa0 = neigh.kneighbors([hgta0rest.flatten()])
distsa0 = distsa0[0]
neighboursa0 = neighboursa0[0]

print("Días más próximos a a0 := {}  :".format(a0))
print()
for i in range(n_neighbours):
    fecha = dt.date(1800, 1, 1) + dt.timedelta(hours= time21[neighboursa0[i]])
    print("Dia {}: {}; Distancia : {}".format(i,fecha,distsa0[i]))

print()
#Calculamos la media de los días más análogos

hgtmedio = np.zeros((len(level),len(latssigma),len(lonssigma)))
airmedio = np.zeros((len(level),len(latssigma),len(lonssigma)))
for i in range(n_neighbours):
    hgtmedio += hgtsigma[neighboursa0[i]]
    airmedio += airsigma[neighboursa0[i]]

hgtmedio /= n_neighbours
airmedio /= n_neighbours

#Calculamos el error absoluto medio de la temperatura

MAE = np.mean(abs(airmedio[level==1000][0] - aira0rest[level==1000][0]))
print("Error absoluto medio de T: {}".format(MAE))

#Dibujamos las gráficas
plt.rcParams.update({'font.size': 7})
    
fig = plt.figure()
fig.subplots_adjust(hspace=0.7, wspace=0.5)

ax = fig.add_subplot(2, 2, 1)
ax.title.set_text('Observación HGT {}-{}-{}'.format(str(a0.day).zfill(2),
                                            str(a0.month).zfill(2),a0.year))

plt.contourf(lonssigma, latssigma, hgta0rest[level == 1000,:,:][0],
             levels = 40, cmap = 'coolwarm')
plt.colorbar()

ax = fig.add_subplot(2, 2, 2)
ax.title.set_text('Selección HGT media (dist. Euclídea)')
plt.contourf(lonssigma, latssigma,hgtmedio[level == 1000,:,:][0], 
             levels = 40,cmap = 'coolwarm')
plt.colorbar()

ax = fig.add_subplot(2, 2, 3)
ax.title.set_text('Observación AIR {}-{}-{}'.format(str(a0.day).zfill(2),
                                              str(a0.month).zfill(2),a0.year))
plt.contourf(lonssigma, latssigma,aira0rest[level == 1000,:,:][0],
             levels = 40, cmap = 'coolwarm')
plt.colorbar()

ax = fig.add_subplot(2, 2, 4)
ax.title.set_text('Predicción AIR por análogos (dist. Euclídea)')
plt.contourf(lonssigma, latssigma,airmedio[level == 1000,:,:][0],
             levels = 40, cmap = 'coolwarm')
plt.colorbar()

fig = plt.gcf()
fig.savefig("obsypredii.pdf",format='pdf')
plt.show()


    



