# -*- coding: utf-8 -*-
"""
Referencias:
    
    Fuente primaria del reanálisis
    https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.pressure.html
    
    Altura geopotencial en niveles de presión
    https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=59&tid=97457&vid=1498
    
    Temperatura en niveles de presión:
    https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=59&tid=97457&vid=4237

    Temperatura en niveles de superficie:
    https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=59&tid=97457&vid=1497
    
"""

import datetime as dt  # Python standard library datetime  module
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
#from scipy.io import netcdf as nc
from sklearn.decomposition import PCA

workpath = ""

#import os
#workpath = "C:/NCEP"
#os.getcwd()
#os.chdir(workpath)
#files = os.listdir(workpath)


f = Dataset(workpath + "air.2021.nc", "r", format="NETCDF4")
#f = Dataset(workpath + "/air.2m.gauss.2021.nc", "r", format="NETCDF4")
#f = nc.netcdf_file(workpath + "/air.2221.nc", 'r')

print(f.history)
print(f.dimensions)
print(f.variables)
time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
level = f.variables['level'][:].copy()
lats = f.variables['lat'][:].copy()
lats_units = f.variables['lat'].units
lons = f.variables['lon'][:].copy()
lons_units = f.variables['lon'].units
air21 = f.variables['air'][:].copy()
air_units = f.variables['air'].units
#air_scale = f.variables['air'].scale_factor
#air_offset = f.variables['air'].add_offset
print(air21.shape)
f.close()

f = Dataset(workpath + "air.2022.nc", "r", format="NETCDF4")
time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
air22 = f.variables['air'][:].copy()
f.close()


f = Dataset(workpath + "hgt.2021.nc", "r", format="NETCDF4")
time21 = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
hgt21 = f.variables['hgt'][:].copy()
hgt_units = f.variables['hgt'].units
#hgt_scale = f.variables['hgt'].scale_factor
#hgt_offset = f.variables['hgt'].add_offset
print(hgt21.shape)
f.close()


#f = nc.netcdf_file(workpath + "/" + files[0], 'r')
f = Dataset(workpath + "hgt.2022.nc", "r", format="NETCDF4")
time22 = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
hgt22 = f.variables['hgt'][:].copy()
f.close()


"""
Ejemplo de evolución temporal de un elemento de aire
"""
plt.plot(time21, air21[:, 0, 0, 0], c='r')
plt.show()


"""
Distribución espacial de la temperatura en el nivel de 1000hPa, para el primer día
"""
plt.contour(lons,lats, air21[0,0,:,:])
plt.show()

hgt21b = hgt21[:,level==1000.,:,:].reshape(len(time21),len(lats)*len(lons))
air21b = air21[:,level==500.,:,:].reshape(len(time21),len(lats)*len(lons))

#air3 = air2.reshape(len(time),len(lats),len(lons))

n_components=4


X = hgt21b
Y = hgt21b.transpose()
pca = PCA(n_components=n_components)
Element_pca0 = pca.fit_transform(Y)
Element_pca0 = Element_pca0.transpose(1,0).reshape(n_components,len(lats),len(lons))
# Supongo que esta (Element_pca0) es la que está bien, aunque reduzca
# el número de días.
# Lo que hace la otra no tiene ningún sentido.

#Interpretar el siguiente resultado
pca.fit(Y)
print(pca.explained_variance_ratio_)
out = pca.singular_values_

#Interpretar el siguiente resultado
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
out = pca.singular_values_

#X = air21b
X = hgt21b
State_pca = pca.fit_transform(X)
Element_pca1 = pca.fit(X).components_
Element_pca1 = Element_pca1.reshape(n_components,len(lats),len(lons))
print(pca.explained_variance_ratio_)

#Ejercicio de la práctica - Opción 1
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    ax.text(0.5, 90, 'PCA-'+str(i),
           fontsize=18, ha='center')
    plt.contour(lons, lats, Element_pca0[i-1,:,:],cmap = 'jet')
    plt.colorbar()
plt.show()

#Ejercicio de la práctica - Opción 2
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    ax.text(0.5, 90, 'PCA-'+str(i),
           fontsize=18, ha='center')
    plt.contour(lons, lats, Element_pca1[i-1,:,:],cmap = 'jet')
    plt.colorbar()
plt.show()



#time_idx = 237  
# Python and the reanalaysis are slightly off in time so this fixes that problem
# offset = dt.timedelta(hours=0)
# List of all times in the file as datetime objects
dt_time21 = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) for t in time21]
np.min(dt_time21)
np.max(dt_time21)

dt_time22 = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) for t in time22]
np.min(dt_time22)
np.max(dt_time22)
dia0 = dt.date(2022, 1, 11)


