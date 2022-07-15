# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:58:33 2020

@author: Robert
"""

#from mpl_toolkits import mplot3d


#from __future__ import division
import os
import numpy as np
from numpy import pi, cos, sin, sqrt, outer, ones
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

vuestra_ruta = ""

os.getcwd()
#os.chdir(vuestra_ruta)


"""
2-esfera
"""
#latitudes
u = np.linspace(0, np.pi, 30)
#longitudes
v = np.linspace(0, 2 * np.pi, 60)

x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))


#Bipunto: 3 coordenadas de origen y 3 coordenadas de dirección
bipunto = np.array([[0, 0, 1, 0.1, 0.1, 0.2], [1, -1, 0, 0.1, 0.1, 0]])

print(*bipunto)
X, Y, Z, U, V, W = zip(*bipunto)


fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.set_zlim(-1,1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.quiver(0.95, -0.3, 0,0,0, 0.7, colors = 'red', zorder=3)
ax.quiver(X, Y, Z, U, V, W, colors="red",zorder=3)
ax.plot_surface(x, y, z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface');
plt.show()



"""
Plano normal a un vector, plano tangente a una esfera
"""

def plano(punto, normal, xlim=(-1,1), ylim=(-1,1), r0=0.1):
    # Si la normal de un plano es es [a,b,c]
    # La ecuación del plano es:  a*x+b*y+c*z+d = 0
    # Entonces sólo nos falta calcular d:
    d = -punto.dot(normal) 
    # creamos una malla de puntos con resolución arbitraria (por ejemplo 0.1)
    xx, yy = np.meshgrid(np.arange(xlim[0], xlim[1], r0), np.arange(ylim[0], ylim[1], r0))
    # Ahora despejamos el valor de zz para obtener la superficie
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
    return xx, yy, zz

"""
Dada una dirección cualquiera, proyección de z
 en el plano tangente a la 2-esfera, en un punto arbitrario
"""
def en_plano(punto, direc):
    normal = punto
    d = -punto.dot(normal) 
    z0 = (-normal[0] * (punto[0]+direc[0]) - normal[1] * (punto[1]+direc[1]) - d) * 1. /normal[2]
    return z0

#El punto sobre la esfera, en coordenadas polares
phi = 0
theta = pi/2
#El punto sobre la esfera, en coordenadas cartesianas
punto = np.array([np.cos(theta)*np.cos(phi),  np.cos(theta)*np.sin(phi), np.sin(theta)])
normal = punto 
#El plano tangente a dicho punto
xx, yy, zz =  plano(punto, punto)

ax = plt.figure().gca(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.plot_surface(x, y, z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,alpha=0.5,
                cmap='viridis', edgecolor='none')
plt.show()




"""
Transporte paralelo de vectores
"""

def transp(theta0, phi, theta):
    phi2 = theta*np.sin((np.sin(theta0))*(phi))/np.cos(theta0)
    theta2 = theta*np.cos((np.sin(theta0))*(phi))
    return phi2, theta2

"""
Vector en la variedad de la 2-esfera
Definimos un punto origen (o) y una dirección (p) 
"""
o_phi = 0   #longitud
o_theta = 0.3*pi/2 #latitud del punto original. Por lo tanto es THETHA0 (!!)

p_phi = 0  #longitud
p_theta = pi/5 #latitud
p_norm = sqrt((p_phi**2)*cos(p_theta)**2 + p_theta**2)
#o_theta = 0

"""
Trasladamos paralelamente el bipunto anterior
"""
Dphi = 0.2*np.pi

o_phi2 = p_phi + Dphi
o_theta2 = o_theta
p_phi2, p_theta2 = transp(theta0=o_theta, phi= Dphi, theta=p_theta)
p_norm2 = sqrt((p_phi2**2)*cos(o_theta2)**2 + p_theta2**2)

"""
CAMBIAMOS EL SISTEMA DE REFERENCIA PARA LA REPRESENTACIÓN
"""
phi0 = np.pi/4

"""
VISTO COMO CURVA PARAMÉTRICA
"""
phi = np.linspace(o_phi, o_phi + p_phi, 100)
theta = np.linspace(o_theta, o_theta + p_theta, 100)
v = np.array([np.cos(theta)*np.cos(phi-phi0),  np.cos(theta)*np.sin(phi-phi0), np.sin(theta)])

phi2 = np.linspace(o_phi2, o_phi2 + p_phi2, 100)
theta2 = np.linspace(o_theta2, o_theta2 + p_theta2, 100)
v2 = np.array([np.cos(theta2)*np.cos(phi2-phi0),  np.cos(theta2)*np.sin(phi2-phi0), np.sin(theta2)])


"""
VISTO COMO BIPUNTO
"""
o = np.array([np.cos(o_theta)*np.cos(o_phi-phi0),  np.cos(o_theta)*np.sin(o_phi-phi0), np.sin(o_theta)])
p = np.array([np.cos(o_theta + p_theta)*np.cos(o_phi + p_phi-phi0),  np.cos(o_theta + p_theta)*np.sin(o_phi + p_phi-phi0), np.sin(o_theta + p_theta)])
X, Y, Z, U, V, W = np.concatenate((o, p-o))
xx, yy, zz =  plano(o, o)

o2 = np.array([np.cos(o_theta2)*np.cos(o_phi2-phi0),  np.cos(o_theta2)*np.sin(o_phi2-phi0), np.sin(o_theta2)])
p2 = np.array([np.cos(o_theta2+p_theta2)*np.cos(o_phi2+p_phi2-phi0),  np.cos(o_theta2 + p_theta2)*np.sin(o_phi2 + p_phi2-phi0), np.sin(o_theta2 + p_theta2)])
X2, Y2, Z2, U2, V2, W2 = np.concatenate((o2, p2-o2))

xx2, yy2, zz2 =  plano(o2, o2)


"""
Curva (el paralelo) donde queremos transladar
"""
phi = np.linspace(0, 2*pi, 100)
theta = np.ones_like(phi)*o_theta
gamma = np.array([np.cos(theta)*np.cos(phi-phi0),  np.cos(theta)*np.sin(phi-phi0), np.sin(theta)])

"""
FIGURA
"""
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)
plt.quiver(X, Y, Z, U, V, W, colors="red",zorder=3, color='red',arrow_length_ratio=0.4)
plt.quiver(X2, Y2, Z2, U2, V2, W2, colors="red",zorder=3, arrow_length_ratio=0.4)

ax.plot(v[0],v[1], v[2], '-b',c="black",zorder=3)
ax.plot(v2[0],v2[1], v2[2], '-b',c="black",zorder=3)
ax.plot(gamma[0], gamma[1], gamma[2], '-b',c="gray",zorder=3)


ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,alpha=0.25,
                cmap='viridis', edgecolor='none')
ax.plot_surface(xx2, yy2, zz2, rstride=1, cstride=1,alpha=0.25,
                cmap='viridis', edgecolor='none')

ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha = 0.6,
                cmap='viridis', edgecolor='none')

plt.show()


#fig.savefig('C:/Users/Rober/Dropbox/Importantes_PisoCurro/Universitat/Profesor Asociado/GCOM/LaTeX/Curvaturas0.png', dpi=300)     # save the figure to file

