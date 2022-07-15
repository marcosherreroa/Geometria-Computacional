# -*- coding: utf-8 -*-
"""
Práctica 1
"""

import os
import matplotlib.pyplot as plt
import numpy as np
#import math as mt


workpath = "C:/"
os.getcwd()
#os.chdir(workpath)
files = os.listdir(workpath)
#print(files)

#Diferentes formas de crear variables vacías (array y listas)
n1 = np.empty([10])
n2 = [0]*10
n3 = [0 for x in range(10)]
print(n1)
print(n2)
print(n3)
print(type(n1))
print(type(n2))
print(type(n3))

"""
OJO CON LA MULTPLICACIÒN DE LISTAS
"""
# -1*[1,2]
# [-1,-1]*[1,2]
a = [-1,-1]
b = [1,2]
print(np.array(a)*np.array(b))
print(np.multiply(a,b))
print([a0*b0 for a0,b0 in zip(a,b)])
print([a[i]*b[i] for i in range(len(a))])
print(map(lambda x,y: x*y, a,b))
#r=2

def logistica(x):
    return r*x*(1-x);

def fn(x0,f,n):
    x = x0
    for j in range(n):
        x = f(x)
    return(x)


def orbita(x0,f,N):
    orb = np.empty([N])
    x = x0
    for i in range(N):
        orb[i] = x
        x = f(x)
    return(orb)
    
    
N = 100
x0 = 0.5
#Probar con r = 0.5; r = 1; r=1.1;  r=2, r=3, r=4
r=0.5
orb = orbita(x0,logistica,N)
#print(orb)
plt.plot(orb)
plt.show()


"""
PODEMOS PINTAR UN PANEL DE VARIOS PLOTS
"""

x0=0.2
fig, axs = plt.subplots(3, 2, figsize=(10, 10))
print(range(3))
rs = [0.5, 1, 1.1, 2, 2.9, 3.1]
for i in range(3):
    for j in range(2):
        r=rs[i*2+j]
        axs[i, j].plot(orbita(x0,logistica,N))
        axs[i, j].title.set_text('r = '+str(r))

#axs[1, 0].scatter(orb)
#axs[0, 1].hist(orb)
#axs[1, 1].hist2d(orb)

plt.show()

"""
DESPUÉS DE LA PRIMERA BIFURCACIÓN!!
"""

N = 50
r=3.1
orb = orbita(x0,logistica,N)
#print(orb)
plt.plot(orb)
plt.show()

"""
Para mostrar los 4 primeros
"""
print(orb[[0,1,2,3]])

"""
Para mostrar los 5 últimos. Ojo con el orden!!
"""
print(orb[np.multiply(-1,[1,2,3,4,5])])
print(orb[np.multiply(-1,np.arange(1,5,1))])
print(orb[-1*np.arange(1,5,1)])
#print(np.arange(10,1,-1))
#print(np.arange(10,0,-1))

"""
Para mostrar los 20 últimos
"""
N=20
ult = orb[-1*np.arange(N,0,-1)]
print(ult)

"""
ANÁLISIS DE LA PERIODICIDAD
"""

epsilon = 0.001

for i in np.arange(2,N,1):
    print(i, "Dist = ", abs(ult[N-1] - ult[N-i]))
    if abs(ult[N-1] - ult[N-i]) < epsilon :
        break
print("Periodo = ",i-1)

"""
Definimos una función que nos devuelva el (posible) periodo de una subórbita dado un épsilon
"""  

def periodo(suborb, epsilon=0.001):
    N=len(suborb)
    for i in np.arange(2,N-1,1):
        if abs(suborb[N-1] - suborb[N-i]) < epsilon :
            break
    return(i-1)
    

"""
Ahora buscamos un conjunto atractor de periodo igual al obtenido
""" 
r = 3.1
#r=3.5
orb = orbita(x0,logistica,80)
ult = orb[-1*np.arange(20,0,-1)]
per = periodo(ult)

#Candidato a conjunto atractor   
#V0 = np.array([ult[N-1-i] for i in range(per)])
V0 = np.sort([ult[N-1-i] for i in range(per)])
print("V0 = " + str(V0)) 
V1 = logistica(V0)  
print("V1 = " + str(V1)) 

Vt = fn(V0, logistica, per)
Dt = np.max(Vt) - np.min(Vt)
print("Vt-V0 = " + str(Vt-V0) + ", Dt = " + str(Dt)) 

V2t = fn(V0, logistica, 2*per)
D2t = np.max(V2t) - np.min(V2t)
print("V2t-V0 = " + str(V2t-V0) + ", D2t = " + str(D2t))
"""
Nótese que el mínimo aumenta y el máximo disminuye: Está acotándose
""" 

M = 2
DIt = 0
for n in np.arange(1,12):
    V2nt = fn(V0, logistica, (2^n)*M*per)
    D2nt = np.max(V2nt) - np.min(V2nt)
    print("D2^(n)t - D2^(n-1)t = " + str(D2nt-DIt))
    DIt = D2nt

print(V2nt)
"""
Depende de x0???

Depende de r??
""" 

"""
FUNCIÓN PARA OBTENER ATRACTORES
""" 


def atrac(f, N0, N, epsilon=0.001):
    orb = orbita(x0,f,N0)
    ult = orb[-1*np.arange(N,0,-1)]
    per = periodo(ult, epsilon)
    V0 = np.sort([ult[N-1-i] for i in range(per)])
    return V0

N0 = 200
N = 50
V0 = atrac(logistica, N0, N,0.0001)

rss = np.arange(3.1,4, 0.0005)
V0s = np.empty([N,len(rss)])*float("nan")

for i in range(len(rss)):
    r = rss[i]
    V0 = atrac(logistica, N0, N)
    V0s[range(len(V0)),i] = V0
    if len(V0)==8 : 
        print(r, V0)
    

plt.figure(figsize=(10,10))
for j in range(N):
    plt.plot(rss, V0s[j,], 'ro', markersize=1)
plt.xlabel = "r"
plt.ylabel = "V0"

plt.axvline(x=3, ls="--")
plt.axvline(x=3.5, ls="--")

plt.show()

plt.figure(figsize=(10,10))
for j in range(N):
    plt.plot(rss[rss > 3.5], V0s[j,rss > 3.5], 'ro', markersize=1)
plt.xlabel = "r"
plt.ylabel = "V0"

plt.axvline(x=3.543, ls="--")
plt.axvline(x=3.564, ls="--")

plt.show()


#print(pd.DataFrame(orb).loc[[0,1,2,3],0])
#print(orb[-[1,2,3]][1,2,3])

# acorr.to_csv(path_or_buf=arch, sep = "\t",index=False, header=False)