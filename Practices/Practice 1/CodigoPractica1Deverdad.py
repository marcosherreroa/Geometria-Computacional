"""
Práctica 1 de Geometría computacional 2021-2022
Autor: Marcos Herrero
"""

import matplotlib.pyplot as plt
import numpy as np

"""
Funciones auxiliares
"""

#Devuelve la funcion logística f de parametro r y evaluada en x
def logistica(x):
    return r*x*(1-x);

#Calcula la sucesion xn hasta encontrar un tiempo transitorio para el que
#la amplitud no varia. Se consideran  conjuntos de inicialmente tamaño m 
#y mayores en cada iteracion, como en el algoritmo de la teoria. 
# Devuelve la orbita obtenida.
def nuevaOrbita(x0,f,m):
    A = 0 #amplitud

    orb = np.empty(m)
    x = x0
    for i in range(m):
        orb[i] = x
        x = f(x)
        
    Aant = A #amplitud del conjunto anterior
    A = np.max(orb) - np.min(orb)
    
    while abs(A-Aant) >= epsilon :
    
        for i in range(m):
            orb = np.append(orb,x)
            x = f(x)
        
        Aant = A
        A = np.max(orb[-m:]) - np.min(orb[-m:])
        
    
    return orb

#Devuelve el periodo de la orbita para el epsilon proporcionado
#Es decir, el numero de iteraciones que separan el ultimo elemento de uno
#con el que difiere menos de epsilon    
def periodo(orb, epsilon=0.001):
    N=len(orb)
    for i in np.arange(2,N-1,1):
        if abs(orb[N-1] - orb[N-i]) < epsilon :
            break
    return(i-1)

#Comprueba que el conjunto limite correspondiente V0
# generado a partir de x0, es estable
def comprobarEstabilidad(x0,per,V0,epsilon,m):
    
    for z in np.arange(-10*epsilon,11*epsilon,epsilon):
        x0mod = x0+z
        orbmod = nuevaOrbita(x0mod,logistica,m)
        Nmod = orbmod.size
        permod = periodo(orbmod,epsilon)
        
        if per != permod:
            return False

        V0mod = np.sort(orbmod[Nmod-permod:])
        dif = np.absolute(V0 - V0mod)
        maxdif = np.max(dif)
          
        if maxdif >= epsilon:
            return False
    
    return True


#Busca un conjunto atractor para r partiendo de x0.
def buscarConjuntoAtractor(r,x0,epsilon):
    
    #1. Calculamos la orbita, de tamaño un tiempo transitorio. Al menos tendra tamaño
    # m = 16 necesario para poder identificar conjuntos atractores
    # de 8 elementos y calcular sus errores

    m = 16
    orb = nuevaOrbita(x0,logistica,m)
    N = orb.size

    #2. Buscamos el periodo de la orbita obtenida y calculamos el conjunto limite
    
    per = periodo(orb,epsilon)
     
    #3. Calculamos los errores de los elementos 
    
    V0 = per*[None]
    for i in range(per):      
        V0[i] = (orb[N-per+i],abs(orb[N-per+i] - orb[N-2*per+i]))
       
    #4. Comprobamos la estabilidad del conjunto limite obtenido

    V0.sort()
    estable = comprobarEstabilidad(x0,per,[e[0] for e in V0],epsilon,m)
    
    return V0, orb, estable
    
"""
Apartado i): Hallar dos conjuntos atractores para r en el rango (3,3.544) 
y x0 en [0,1]

"""

print("Nota: el redondeo de acuerdo a las cifras significativas se hace "
      "posteriormente a mano")


print()
print("Apartado i):")
#Primer conjunto atractor
r = 3.4
x0 = 0.5
epsilon = 0.001
print("Primer conjunto atractor: r = {}, x0 = {}".format(r,x0))
print()
V0, orb, estable = buscarConjuntoAtractor(r,x0,epsilon)

if not(estable):
    print("No se encontró un conjunto estable")

else:
    for (x,delta) in V0:
        print("{} +- {}".format(x,delta))
    
    
    plt.title("Conjunto atractor r = {}".format(r))
    plt.plot(orb)
    plt.xlabel("Iteración")
    plt.ylabel("x")
    fig = plt.gcf()
    fig.savefig("conjuntoi1.pdf",format='pdf')
    plt.show()
    
        
print()
print()

#Segundo conjunto atractor
r = 3.3
x0 = 0.2
epsilon = 0.001
print("Segundo conjunto atractor: r = {}, x0 = {}".format(r,x0))
print()
V0,orb, estable = buscarConjuntoAtractor(r,x0,epsilon)


if not(estable):
    print("No se encontró un conjunto estable")

else:
    for (x,delta) in V0:
        print("{} +- {}".format(x,delta))
    
    
    plt.title("Conjunto atractor r = {}".format(r))
    plt.plot(orb)
    plt.xlabel("Iteración")
    plt.ylabel("x")
    fig = plt.gcf()
    fig.savefig("conjuntoi2.pdf",format='pdf')
    plt.show()
    

print()
print()

"""
Apartado ii): Estimar los valores de r en (3.544, 4) para los que el conjunto
atractor tiene 8 elementos
"""  


print("Apartado ii)")

x0 = 0.5
epsilon = 0.001
rerror = 0.001# error en la r

rs = np.arange(3.544,4,rerror)
rs8el = []
ejemplo = None
rejemplo = 0

xeje = []
yeje = []
xejeesp = []
yejeesp = []

for r in rs:
    V0,orb, _ = buscarConjuntoAtractor(r, x0, epsilon)
    conj = [V0[i][0] for i in range(len(V0))]
    
    for elem in conj:
        xeje.append(r)
        yeje.append(elem)
        

    if len(V0) == 8:
        rs8el.append(r)
        
        for elem in conj:
            xejeesp.append(r)
            yejeesp.append(elem)
        
        if ejemplo == None:
            ejemplo = V0
            rejemplo = r
            
            plt.title("Conjunto atractor r = {}".format(round(r,3)))
            plt.plot(orb)
            plt.xlabel("Iteración")
            plt.ylabel("x")
            fig = plt.gcf()
            fig.savefig("conjuntoii.pdf",format='pdf')
            plt.show()
     
#print(V0s)

plt.title("Valores de r con con periodo 8")
plt.plot(xeje,yeje,'b,',markersize=0.01)
plt.plot(xejeesp,yejeesp,'ro', markersize=2)
plt.xlabel("r")
plt.ylabel("x")
fig = plt.gcf()
fig.savefig("rs8elems.pdf",format='pdf')
plt.show()

print("Los r encontrados para los que el conjunto atractor tiene 8 elementos son:")
for r in rs8el:
    print("{} +- {}".format(r,rerror))
    
print()

print("Un ejemplo de conjunto atractor (r = {}):".format(rejemplo))
for (x,delta) in ejemplo:
    print("{} +- {}".format(x,delta))
