# -*- coding: utf-8 -*-
"""
Práctica 2

Autor : Marcos Herrero
"""

from collections import Counter
import numpy as np
import pandas as pd
import math


#Fusiona los dos estados con menor frecuencia para obtener una nueva rama
def huffmanBranch(distrib):
    #Extraemos los estados con menor frecuencia, que vamos a fusionar
    toMerge = distrib.head(2)
    distrib.drop(index = [0,1], inplace=True)
    
    #Anotamos el codigo de los estados fusionados
    codigo = np.array([toMerge['states'][0], toMerge['states'][1]])

    #Fusionamos los estados, calculando la probabilidad del nuevo
    newState = ''.join(np.array(toMerge['states']))
    newProb = np.sum(toMerge['probab'])
    
    #Añadimos el nuevo estado a la distribucion
    df = pd.DataFrame({'states' : [newState], 'probab' : [newProb]})
    distrib = pd.concat([distrib, df])
    
    #Reordenamos de menor a mayor frecuencia
    distrib = distrib.sort_values(by='probab', ascending=True)
    distrib.index=np.arange(0,len(distrib.index))
    
    return distrib, codigo

#Devuelve el codigo de Huffman del lenguaje y su longitud media
def huffmanCode (filename):
    
    with open(filename, 'r',encoding="utf8") as file:
        text = file.read()
     
    # Calculamos las frecuencias de aparición de cada carácter en el texto
    counter = Counter(text)
    states = np.array(list(counter))
    weights = np.array(list(counter.values()))
    probs = weights/float(np.sum(weights))
    distrib = pd.DataFrame({'states': states, 'probab': probs})
    
    # Calculamos, a partir de las frecuencias, la entropía del lenguaje
    H= 0
    for frec in probs:
        H += (-frec*math.log2(frec))
    
    #Ordenamos los caracteres de menor a mayor frecuencia
    distrib = distrib.sort_values(by='probab',ascending=True)
    distrib.index = np.arange(0,len(states))
    
    #Calculamos el orden en que los nodos se van fusionando, junto
    #con los dígitos que va adquiriendo cada uno
    #(que es una forma de representar el árbol de Huffman)
    
    ordenFusion = np.array([])
    
    while len(distrib) > 1:
        distrib, codigo = huffmanBranch(distrib)
        #Los elementos de las posiciones pares adquieren un 0 en esa rama,
        #y los de las posiciones impares un 1
        ordenFusion = np.concatenate((ordenFusion, codigo), axis=None)
    
    
    #Calculamos el código de Huffman del lenguaje
    
    codigoHuffman = {}
    
    
    for caract in states:
        codigoHuffman[caract] = []
    
    for i in range(len(ordenFusion)//2):
        
        for caract in ordenFusion[2*i]:
                
            codigoHuffman[caract].append('0')
        
        for caract in ordenFusion[2*i+1]:
            
            codigoHuffman[caract].append('1')
    
    #Obsérvese que los códigos se construyen al reves, asi que
    # hay que darles la vuelta. Aprovechamos para escribirlos como string
    
    for caract in states:
        codigoHuffman[caract] = ''.join(reversed(codigoHuffman[caract]))
    
    #Calculamos la longitud media de los codigos de Huffman (ponderada por frecuencia)
    L = 0
    for i in range(len(states)):
        L += (len(codigoHuffman[states[i]])*probs[i])
    
    
    return codigoHuffman,L,H

# Devuelve la codificación de la palabra X en binario usual UTF-8
def codificarBinarioUsual (X):
    CodXBin = bytes(X,'utf8')
    CodXBin = [str(bin(x))[0]+str(bin(x))[2:] for x in CodXBin]
    CodXBin = ''.join(CodXBin)
    
    return CodXBin

# Devuelve la codificación de la palabra X con el código que se pasa 
# como parámetro
def codificar (X,code):
    CodX = []

    for caract in X:
        CodX.append(code[caract])

    CodX = ''.join(CodX)
    
    return CodX

# Devuelve la codificación de la palabra CodX con el código inverso que se pasa
# como parámetro
def decodificar (CodX,revcode):
    X = []
    i , j = 0, 1

    while i < len(CodX):
        while j <= len(CodX) and CodX[i:j] not in revcode:
            j+=1
            
        if j > len(CodX):
            print("La palabra no forma parte del lenguaje")
            break
            
        X.append(revcode[CodX[i:j]])
        
        i = j
        j = i+1

    X = ''.join(X)
    
    return X

'''
Apartado i) : hallar el código de Huffman de Seng y Sesp y sus longitudes medias
y comprobar que se cumple el teorema de Shannon
'''

print('Apartado i)')
print()
#Obtenemos el códigos de huffman de las muestras, la longitud media 
#y la entropía del inglés


huffCodeEn, LEn, HEn = huffmanCode('GCOM2022_pract2_auxiliar_eng.txt') 

print("Lenguaje Seng:")
print("==============")
print("Codigo Huffman:")

for key,value in sorted(huffCodeEn.items(),key=lambda x:x[1]):
    print(" {} : {}".format(key,value))
print()
   
print("Longitud media: {}".format(LEn))
print("Entropía : {}".format(HEn))
print()
print()
  
#Obtenemos el códigos de huffman de las muestras, la longitud media 
#y la entropía del español
huffCodeEsp, LEsp, HEsp = huffmanCode('GCOM2022_pract2_auxiliar_esp.txt')

print("Lenguaje Sesp:")
print("==============")
print("Codigo Huffman:")

for key,value in sorted(huffCodeEsp.items(),key=lambda x:x[1]):
    print(" {} : {}".format(key,value))
print()

print("Longitud media: {}".format(LEsp))
print("Entropía : {}".format(HEsp))
print()
print()



'''
Apartado ii) : codificar la palabra medieval y comparar la longitud del
código obtenido con la longitud que habría sido necesaria en binario usual.
Comparar tb la eficiencia en general.
'''

print('Apartado ii)')
print()

X = 'medieval'

print("Palabra a tratar : {}".format(X))
print()

#Con binario usual (UTF-8)

CodXBin = codificarBinarioUsual(X)

print("Codificación de la palabra con binario usual (UTF-8): {}".format(CodXBin))
print("Longitud: {}".format(len(CodXBin)))
print()

#Con Seng
CodXEng = codificar(X,huffCodeEn)

print("Codificación de la palabra con Seng: {}".format(CodXEng))
print("Longitud: {}".format(len(CodXEng)))
print()

#Con Sesp
CodXEsp = codificar(X,huffCodeEsp)

print("Codificación de la palabra con Sesp: {}".format(CodXEsp))
print("Longitud: {}".format(len(CodXEsp)))
print()

'''
Apartado iii) : decodificar la palabra 10111101101110110111011111 con Seng
'''

print('Apartado iii)')
print()

CodYEng = '10111101101110110111011111'

print('Codificación a tratar : {}'.format(CodYEng))


#Canstruimos el hashmap inverso de huffCodeEng (la relación es biyectiva)
revHuffCodeEng = dict(reversed(item) for item in huffCodeEn.items())

#Decodificamos con Seng
Y = decodificar(CodYEng,revHuffCodeEng)

print("Decodificación: {}".format(Y))
print()
print()

'''
Codificar y decodificar una misma palabra con ambos lenguajes
'''

print('Codificar y decodificar una misma palabra:')
print()

#Con SEng
palabra = 'dragon'
print("Con SEng")
print("Palabra elegida: {}".format(palabra))

cod = codificar(palabra,huffCodeEn)
print("Codificada: {}".format(cod))

redecod = decodificar(cod,revHuffCodeEng)
print("Redecodificada: {}".format(redecod))
print()

#Con SEsp
palabra = 'queso'
print("Con SEsp")
print("Palabra elegida: {}".format(palabra))

cod = codificar(palabra,huffCodeEsp)
print("Codificada: {}".format(cod))

revHuffCodeEsp = dict(reversed(item) for item in huffCodeEsp.items())
redecod = decodificar(cod,revHuffCodeEsp)
print("Redecodificada: {}".format(redecod))