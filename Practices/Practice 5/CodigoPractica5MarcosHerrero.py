# -*- coding: utf-8 -*-
"""
Práctica 5 de Geometría Computacional
Autor: Marcos Herrero
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

'''
Funciones auxiliares
'''

#Realiza la proyección estereográfica de la esfera con el alpha dado.
#Devuelve las coordenadas cartesianas proyectadas de la esfera (xs,ys,zs) sobre el plano z = 0
def proyEstereo(xs,ys,zs,alpha):
    xsproy = np.divide(xs,abs(1-zs)**alpha, out = np.zeros(xs.shape),where = zs != 1)
    ysproy =  np.divide(ys,abs(1-zs)**alpha, out = np.zeros(ys.shape), where = zs != 1)
    zsproy = np.full(zs.shape,0)
    
    return xsproy,ysproy,zsproy

def fAnimation(t,xs,ys,zs):
    xstcurva = 2*xs/(2*(1-t)+(1-zs)*t)
    ystcurva = 2*ys/(2*(1-t)+(1-zs)*t)
    zstcurva = -t+zs*(1-t)

    ax = plt.axes(projection='3d')
    ax.set_zlim3d(-1,1)
    ax.plot_surface(xstcurva, ystcurva, zstcurva, rstride=1, cstride=1, alpha=0.5,
                    cmap='viridis', edgecolor='none')
    

'''
Apartado i): Estimar y representar una malla regular de puntos de S1.
Estimar y representar la imagen de la proyección estereográfica
Diseñar una curva sobre S1 para comprobar cómo se deforma.
'''

#Coordenadas esfericas segun las restricciones pedidas
lats = np.linspace(0.1,np.pi,30)
lons = np.linspace(0,2*np.pi,60)

#Las pasamos a coordenadas cartesianas
xs = np.outer(np.sin(lats), np.sin(lons))
ys = np.outer(np.sin(lats), np.cos(lons))
zs = np.outer(np.cos(lats), np.ones_like(lons))

#Diseñamos una curva sobre la esfera
ts = np.linspace(-1.0, 1.0, 200)
xscurva = np.cos(2*ts)/np.sqrt(1+4*ts**2)
yscurva = -np.sin(2*ts)/np.sqrt(1+4*ts**2)
zscurva = np.sqrt(1-xscurva**2-yscurva**2)


 
#Representamos la esfera con una curva sobre ella
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.plot_surface(xs, ys, zs, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.plot(xscurva, yscurva, zscurva, '-b',zorder=3)
fig.savefig("Esfera.pdf",format='pdf')
plt.show()

#Proyectamos sobre el plano z= 0 mediante proy. estereo. con alpha = 0.5
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection ='3d')
alpha = 0.5
xsproy, ysproy, zsproy = proyEstereo(xs,ys,zs,alpha)
xscurvproy, yscurvproy, zscurvproy = proyEstereo(xscurva,yscurva,zscurva,alpha)
ax = plt.axes(projection='3d')
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.plot_surface(xsproy, ysproy,zsproy,rstride=1, cstride=1,
                cmap='viridis',alpha =0.5, edgecolor='purple')
ax.plot(xscurvproy,yscurvproy,zscurvproy, '-b',zorder=3)
fig.savefig('EsferaProyectada.pdf',format='pdf')
plt.show()

'''
Apartado ii): Obtener una animación de al menos 20 fotogramas de la familia paramétrica dada
'''

fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, fAnimation,np.arange(0,1.05,0.05),
                              fargs = (xs,ys,zs),interval = 20)
ani.save("animación.gif", fps = 5) 
