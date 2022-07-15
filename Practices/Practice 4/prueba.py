# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 15:07:46 2022

@author: marco
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

X = pd.DataFrame([[1,1,1],[1,0,0],[1,1,0],[0,1,1],[0,0,1],[0,0,0]])
y = np.array([0,2,0,1,1,2])
target_names = ['N','M','S']

pca = PCA(n_components=3)
X_r = pca.fit(X).transform(X)


# Percentage of variance explained for each components
print(
    "explained variance ratio (first two components): %s"
    % str(pca.explained_variance_ratio_)
)

print("Components: {}".format(pca.components_))
print("Variance {}".format(pca.explained_variance_))
print("Singular values : {}".format(pca.singular_values_))

X_c = X.apply(lambda x: (x - x.mean()))
print("X_c : {}".format(X_c))

mat = np.dot(X_c.transpose(), X_c)

print("X^TX : {}".format(mat))

w,v = np.linalg.eig(mat)
print("Eigenvalues : {} {}".format(w,v))

"""
Entonces:

explained_variance parece ser los autovalores mayores de X^TX/(n-1)
singular_values son las raíces cuadradas de los autovalores de X^TX (?)

components deberían ser los autovectores de X^TX y sus proporcionales, pero 
parece que hay inestabilidad con el tercero ...

transform devuelve a forma reducida de X (el resumen?)

En vd no entiendo nada

"""

print("X_r: {}".format(X_r))

plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2


'''
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of IRIS dataset")

plt.show()
'''