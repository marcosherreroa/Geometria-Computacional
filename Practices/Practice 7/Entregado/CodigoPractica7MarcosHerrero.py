# -*- coding: utf-8 -*-
"""
Práctica 7 de Geometría Computacional
Autor: Marcos Herrero
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from scipy.spatial import ConvexHull
from skimage import io


'''
Funciones auxiliares
'''

#Obtiene el centroide del sistema S
def calcularCentroide(S):
    return np.mean(S,axis=0)

#Obtiene el diametro del sistema 3D (hay que pasar solo las variables de estado espaciales)
def calcularDiam(Sesp):
    hull = ConvexHull(Sesp)
    diam = 0
    
    for i in range(len(hull.vertices)):
        for j in range(i+1,len(hull.vertices)):
            dist = np.linalg.norm(Sesp[hull.vertices[i],:]- Sesp[hull.vertices[j],:])
            if dist > diam:               
                diam = dist
   
    return diam


#Transformación del sistema S cuyas filas representan los elementos del sistema
#y cuyas columnas representan sus variables de estado. La transformación realizada
# es P' = C + M(P-C) + v 
def transf(S,C,M,v):
    return (np.atleast_2d(C).T + np.matmul(M,(S-C).T)+ np.atleast_2d(v).T).T

#Realiza la rotación de theta grados en el plano XY (con centro en C) y traslación de vector (d,d,0,...0)
#Las variables X e Y han de ser las dos primeras columnas del sistema
def rotaTrasla(S,C,theta,d):
    M = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    v = np.array([d,d])
    
    return np.concatenate([transf(S[:,:2],C[:2],M,v),S[:,2:]],axis = 1)

#Animación utilizando contour
def fAnimationContour(t,S,theta,C,d,shape):    
    St = rotaTrasla(S,C,theta*t,d*t)
    ax = plt.axes(projection='3d')
    ax.contour(St[:,0].reshape(shape),St[:,1].reshape(shape),St[:,2].reshape(shape),
               16,extend3d=True,cmap = plt.cm.get_cmap('coolwarm'))

#Animación utilizando scatter
def fAnimationScatter(t,S,theta,C,d):
    St = rotaTrasla(S,C,theta*t,d*t)
    ax = plt.axes(projection='3d')
    color = list(zip(St[:,3]/256,St[:,4]/256,St[:,5]/256))
    ax.scatter3D(St[:,0],St[:,1],St[:,2], c = color,animated=True)



theta = 3*np.pi

'''
Apartado i): Partiendo de la figura dada, realizar una animación de una familia paramétrica continua que reproduzca
desde la identidad hasta rotación de ángulo 3pi + traslación con v = (d,d,0)
'''

print("Apartado i)")
print("============")


#Figura dada
X, Y, Z = axes3d.get_test_data(0.05)
shape = Z.shape

#Formamos el sistema
#El sistema tiene 3 variables de estado: las coordenadas espaciales X,Y,Z
#(el color es dependiente de Z)
S = np.column_stack([X.flatten(),Y.flatten(),Z.flatten()])
C = calcularCentroide(S) #centroide
d = calcularDiam(S) #diámetro

print("Centroide: {}".format(C))
print("Diámetro: {}".format(d))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour(X, Y, Z, 16, extend3d=True,cmap = plt.cm.get_cmap('coolwarm'))
fig.savefig('fig1ini.pdf',format='pdf')
plt.show()

#Posición final
S1 = rotaTrasla(S, C, theta, d)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour(S1[:,0].reshape(shape),S1[:,1].reshape(shape),S1[:,2].reshape(shape),
           16,extend3d=True,cmap = plt.cm.get_cmap('coolwarm'))
fig.savefig('fig1fin.pdf',format='pdf')
plt.show()

#Animación
fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, fAnimationContour,frames=np.arange(0,1.01,0.05),
                              fargs = (S,theta,C,d,shape), interval=20)
ani.save('animacion.gif',fps=5)
plt.show()


'''
Parte adicional: probamos la misma transformación con otra figura, en la que el color sí representa
dimensiones adicionales
'''


print('Parte adicional:')
print('=============')
#Generamos un cilindro sobre el que aplicaremos la transformación


r = 1
phiini = np.linspace(0,2*np.pi,60)
xini = np.linspace(0,5,20)
phi, x = np.meshgrid(phiini,xini)

y = r*np.cos(phi)
z = r*np.sin(phi)

X = x.flatten()
Y = y.flatten()
Z = z.flatten()

rgb = plt.cm.get_cmap('inferno')(plt.Normalize()(X+Z))[:,:3]
R = rgb[:,0]*256 # rango (0-255) para los colores
G = rgb[:,1]*256
B = rgb[:,2]*256




#Formamos el sistema. Tiene 6 variables de estado: las 3 espaciales y RGB
S = np.column_stack([X.flatten(),Y.flatten(),Z.flatten(),R.flatten(),G.flatten(),B.flatten()])
C = calcularCentroide(S) #centroide
d = calcularDiam(S[:,0:3]) #diámetro (solo var espaciales)

print("Centroide: {}".format(C))
print("Diámetro: {}".format(d))

fig = plt.figure()
ax = plt.axes(projection='3d')
color = list(zip(S[:,3]/256,S[:,4]/256,S[:,5]/256))
ax.scatter3D(X, Y, Z, c = color)
ax.plot3D(C[0],C[1],C[2],'*')
fig.savefig('fig2ini.pdf',format='pdf')
plt.show()

#Posición final
S1 = rotaTrasla(S,C,theta,d)

fig = plt.figure()
ax = plt.axes(projection='3d')
color = list(zip(S1[:,3]/256,S1[:,4]/256,S1[:,5]/256))
ax.scatter3D(S1[:,0], S1[:,1], S1[:,2], c = color)
fig.savefig('fig2fin.pdf',format='pdf')
plt.show()

#Animación

fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, fAnimationScatter,frames=np.arange(0,1.01,0.05),
                              fargs = (S,theta,C,d), interval=20)
ani.save('animacionExtra.gif',fps=5)
plt.show()


'''
Apartado ii)
'''

print('Apartado ii)')
print('=============')

#Cargamos la imagen
img = io.imread('arbol.png')

#Construimos el sistema. Tiene 7 variables de estado : las 3 coordenadas espaciales
# y 4 para construir el color en forma rgba
x = np.arange(0,img.shape[0])
y = np.arange(0,img.shape[1])
X,Y = np.meshgrid(x,y)
X = X.flatten()
Y = Y.flatten()
Z = np.zeros(X.size) #añadimos una dimensión para trabajar en el espacio tridimensional

R = img[:,:,0].flatten()
G = img[:,:,1].flatten()
B = img[:,:,2].flatten()
A = img[:,:,3].flatten()

S = np.column_stack([X,Y,Z,R,G,B,A])

#Representamos el estado inicial
fig = plt.figure()
color = list(zip(R/256,G/256,B/256,A/256))
plt.scatter(X,Y,c=color)
fig.savefig('ap2ini.pdf',format = 'pdf')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(S[:,0],S[:,1],S[:,2],c = color)
plt.show()

#Nos quedamos con el subsistema sigma para el que el color rojo es menor que 240
sigma = S[np.where(S[:,3] < 240)]

C = calcularCentroide(sigma) #centroide
d = calcularDiam(sigma[:,0:2]) #diámetro

fig = plt.figure()
ax = plt.axes(projection ='3d')
color = list(zip(sigma[:,3]/256,sigma[:,4]/256, sigma[:,5]/256, sigma[:,6]/256))
ax.scatter3D(sigma[:,0],sigma[:,1], sigma[:,2],c=color)
fig.savefig('sigmaini.pdf',format = 'pdf')
plt.show()


print("Centroide: {}".format(C))
print("Diámetro: {}".format(d))

#Posición final
sigma1 = rotaTrasla(sigma,C,theta,d)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(sigma1[:,0],sigma1[:,1], sigma1[:,2],c=color)
fig.savefig('sigmafin.pdf',format='pdf')
plt.show()

#Animación

fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, fAnimationScatter,frames=np.arange(0,1.01,0.05),
                              fargs = (sigma,theta,C,d), interval=20)
ani.save('animacion2.gif',fps=5)

 



