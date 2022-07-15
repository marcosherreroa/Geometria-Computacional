# -*- coding: utf-8 -*-
"""
Práctica 8 de Geometría Computacional
Autor: Marcos Herrero
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#Transforma de coordenadas esféricas a cartesianas
def esfToCart(phi,theta):
    x = np.cos(theta)*np.cos(phi)
    y = np.cos(theta)*np.sin(phi)
    z = np.sin(theta)
    
    return np.array([x,y,z])

#Obtiene las coordenadas cartesianas del punto del plano tangente de o determinado
# por el vector v tangente a la esfera en o
def puntoPlano(o,o_phi,o_theta,v_phi,v_theta):
    b1 = np.array([-np.cos(o_theta)*np.sin(o_phi),np.cos(o_theta)*np.cos(o_phi),0])
    b2 = np.array([-np.sin(o_theta)*np.cos(o_phi),-np.sin(o_theta)*np.sin(o_phi),np.cos(o_theta)])
    
    return o + v_phi * b1 + v_theta * b2

#Familia paramétrica pedida en el apartado i)
def famParam(t,phi,theta0,v02):
    p_phi = v02/np.cos(theta0) * np.sin(np.sin(theta0)*phi*t**2)
    p_theta = v02* np.cos(np.sin(theta0)*phi*t**2)
    return p_phi,p_theta


#Función que realiza la animación
def fAnimation(t,theta01,theta02,v02, xesf,yesf,zesf):
    ax = plt.axes(projection = '3d')
    
    o_phi1 = 2*np.pi*t**2
    o_theta1 = theta01
    v_phi1,v_theta1 = famParam(t,2*np.pi,theta01,c0)
    
    o1 = esfToCart(o_phi1,o_theta1)
    p1 = puntoPlano(o1, o_phi1,o_theta1,v_phi1,v_theta1)
    X1,Y1,Z1,U1,V1,W1 = np.concatenate((o1,p1-o1))
    
    o_phi2 = 2*np.pi*t**2
    o_theta2 = theta02
    v_phi2,v_theta2 = famParam(t,2*np.pi,theta02,c0)
    
    o2 = esfToCart(o_phi2,o_theta2)
    p2 = puntoPlano(o2, o_phi2,o_theta2,v_phi2,v_theta2)
    X2,Y2,Z2,U2,V2,W2 = np.concatenate((o2,p2-o2))
    
    ax.plot_surface(xesf, yesf, zesf, cmap='viridis', edgecolor='none',alpha = 0.55)
    ax.quiver(X1,Y1,Z1,U1,V1,W1,colors="blue", zorder=3, arrow_length_ratio=0.4)
    ax.quiver(X2,Y2,Z2,U2,V2,W2,colors="red", zorder=3, arrow_length_ratio=0.4)
    
    

v02 = np.pi/5

#Sistema de referencia
phi = np.linspace(0, 2*np.pi, 30)
theta = np.linspace(-np.pi/2, np.pi/2, 20)
r = 1

x =r* np.outer(np.cos(theta), np.cos(phi))
y = r* np.outer(np.cos(theta), np.sin(phi))
z = r* np.outer(np.sin(theta), np.ones_like(phi))

c0 = np.sqrt(np.pi/5)




'''
Apartado ii): Realizar una animación de la transformación anterior de forma que dos 
copias de v0 se trasladen en 2 paralelos diferentes
'''

theta01 = 0
theta02 = np.pi/6

fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, fAnimation,frames=np.arange(0,2.01,0.05),
                              fargs = (theta01,theta02,v02,x,y,z), interval=20)
ani.save('animacion.gif',fps=5)
plt.show()


