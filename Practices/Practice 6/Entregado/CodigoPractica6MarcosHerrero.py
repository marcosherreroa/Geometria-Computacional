# -*- coding: utf-8 -*-
"""
Práctica 6
Geometría Computacional
Autor: Marcos Herrero
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial import ConvexHull, convex_hull_plot_2d

'''
Funciones auxiliares

'''

#Funcion que define el sistema
def FSistema(q):
    return -2*q*(q*q-1)

#Devuelve los primeros n elementos de la órbita q(t) solución de dq = F(q) 
# con condiciones iniciales q0 y dq0 y granularidad de tiempo delta
def orb(n,F,q0,dq0,delta):
    q = np.empty(n)
    q[0] = q0
    q[1] = q0 + delta*dq0
    
    for i in range(2,n):
        q[i] = delta**2*F(q[i-2])-q[i-2]+2*q[i-1]
    
    return q
 
def obtenDt(D0,t,delta):
    n = int(t/delta)
    Dt = []
    
    for q0,p0 in D0:
        dq0 = 2*p0
        
        #Calculamos la componente q de la órbita (calculamos un elemento más para poder derivar)
        q = orb(n+1,FSistema, q0, dq0,delta)
        
        #Calculamos su derivada discreta
        dq = (q[1:n+1]-q[0:n])/delta
        
        #A partir de la derivada se obtiene (por Hamilton-Jacobi) la componente p de la curva
        p = dq/2
        
        #Eliminamos el último elemento de q
        q = q[0:n]
        
        #Añadimos a Dt el punto obtenido
        Dt.append([q[-1],p[-1]])
    
    return Dt

#Obtener el área de la distribución de fases Dt para t= 1/4 y el delta dado
def obtenareaDt(q0s,p0s,delta):
    t = 1/4
    m1 = len(q0s)
    m2 = len(p0s)
    
    #Calculamos el área de la envolvente convexa de Dt
    D0 = [[q0s[i],p0s[j]] for i in range(m1) for j in range(m2)]
    Dt = obtenDt(D0,t,delta)
    DtHull = ConvexHull(Dt)
    areaDtHull = DtHull.volume
    
    #Calculamos el área resultado de transformar el segmento p = 0
    aristainf_0 = [[q0s[i],p0s[0]] for i in range(m1)]
    aristainf_t = obtenDt(aristainf_0,t,delta)
    hullaristainf_t = ConvexHull(aristainf_t) 
    areaaristainfHull = hullaristainf_t.volume
    
    #Calculamos el área resultado de transformar la recta q = 1
    aristader_0 = [[q0s[m1-1],p0s[i]] for i in range(m2)]
    aristader_t = obtenDt(aristader_0,t,delta)
    hullaristader_t = ConvexHull(aristader_t) 
    areaaristaderHull = hullaristader_t.volume
    
    #El área real de Dt es la de su cobertura convexa menos el excedente debido a
    #a la curvatura de las aristas p=0 y q=1
    return areaDtHull - areaaristainfHull - areaaristaderHull

#Realiza la animación pedida
def fAnimation(t,D,m1,m2):
    ax = plt.axes()
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    
    for q,p in D[t]:
        ax.plot(q,p,marker ='o',markerfacecolor = 'tab:blue',markeredgecolor = 'tab:blue')
      
'''
Apartado i): Representar gráficamente el espacio fásico de las órbitas finales del 
del sistema (D(0,inf)). Nos guardamos además los puntos que calculamos para usarlos
en los siguientes apartados
'''
delta = 10**(-3)  #granularidad del tiempo
n = int(10/delta) #numero de puntos de cada órbita
m1 = 20
m2 = 20

ts = np.arange(n)*delta
D = [[] for i in range(n)]

#Condiciones iniciales (D0) de todo el espacio de fases
q0s = np.linspace(0,1,m1)
p0s = np.linspace(0,1,m2)

fig = plt.figure()
ax = fig.add_subplot(111)

#Representamos una curva por cada una de las condiciones iniciales anteriores
for i in range(m1):
    for j in range(m2):
        color = (1+i+j*(len(q0s)))/(len(q0s)*len(p0s))
        
        #Condiciones iniciales de la componente q de la orbita
        q0 = q0s[i]
        dq0 = 2*p0s[j]
        
        #Calculamos la componente q de la órbita (calculamos un elemento más para poder derivar)
        q = orb(n+1,FSistema, q0, dq0,delta)
        
        #Calculamos su derivada discreta
        dq = (q[1:n+1]-q[0:n])/delta
        
        #A partir de la derivada se obtiene (por Hamilton-Jacobi) la componente p de la curva
        p = dq/2
        
        #Eliminamos el último elemento de q
        q = q[0:n]
        
        
        #Representamos la curva
        ax.plot(q,p,c=plt.get_cmap('winter')(color),lw = 0.5)
        
        
        #Guardamos los puntos para reutilizarlos en los apartados posteriores
        for k in range(n):
            D[k].append([q[k],p[k]])

plt.xlabel('q(t)')
plt.ylabel('p(t)')
fig.savefig('espacioFasico.pdf',format='pdf')
plt.show()

'''
Apartado ii): Obtener el área de D_t para t=1/4 y su intervalo de error. Comprobar
además si se cumple el teorema de Liouville
'''

t = 1/4

#Área de D0: es exacta porque lo hemos construido nosotros explícitamente
delta = 10**(-3)

hullD0 = ConvexHull(D[0])
ax = plt.axes(xlabel = 'q(t)',ylabel = 'p(t)')
fig = convex_hull_plot_2d(hullD0,ax)
fig.savefig('D0.pdf',format='pdf')
plt.show()

areaD0 = hullD0.volume

print('Área de D0: {:.5f}'.format(areaD0))


#Área de Dt: requiere aproximarse
delta = 10**(-3)

hullDt = ConvexHull(D[int(t/delta)])
ax = plt.axes(xlabel = 'q(t)',ylabel = 'p(t)')
fig = convex_hull_plot_2d(hullDt,ax)
fig.savefig('Dt.pdf',format='pdf')
plt.show()

aristainf_0 = [[q0s[i],p0s[0]] for i in range(m1)]
aristainf_t = obtenDt(aristainf_0,t,delta)
hullaristainf_t = ConvexHull(aristainf_t)
ax = plt.axes(xlabel = 'q(t)',ylabel = 'p(t)')
fig = convex_hull_plot_2d(hullaristainf_t,ax)
fig.savefig("AristaInf_t.pdf",format='pdf')
plt.show()


aristader_0 = [[q0s[m1-1],p0s[i]] for i in range(m2)]
aristader_t = obtenDt(aristader_0,t,delta)
hullaristader_t = ConvexHull(aristader_t)
ax = plt.axes(xlabel = 'q(t)',ylabel = 'p(t)')
fig = convex_hull_plot_2d(hullaristader_t,ax)
fig.savefig('AristaDer_t.pdf',format='pdf')
plt.show()

areaDt = hullDt.volume - hullaristainf_t.volume - hullaristader_t.volume

areaDtant = -1
error = -1
errorant = -1

while errorant == -1 or error/errorant >= 0.5:
    delta /= 2
    areaDtant = areaDt
    areaDt = obtenareaDt(q0s,p0s,delta)
    
    errorant = error
    error = abs(areaDt-areaDtant)
 
print("Área de Dt: {:.5f} | Error: {:.5e}".format(areaDt,error)) 


'''
Apartado iii): Realizar una animación de D_t para t entre 0 y 5
'''

'''
delta = 10**(-3)

fig = plt.figure(figsize = (10,10))
ani = animation.FuncAnimation(fig, fAnimation,range(0,int(5//delta),200),
                              fargs = (D,m1,m2),interval = 20)
ani.save("animación.gif", fps = 5) 
'''

