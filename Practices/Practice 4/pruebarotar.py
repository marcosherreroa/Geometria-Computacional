# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:24:31 2022

@author: marco
"""

import datetime as dt  # Python standard library datetime  module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from sklearn.decomposition import PCA

arr = np.empty([6,5,4,3])

for i1 in range(6):
    for i2 in range(5):
        for i3 in range(4):
            for i4 in range(3):
                arr[i1][i2][i3][i4] = 60*i1+12*i2+3*i3+i4
        
print(arr)
lons = np.array([0, 180, 350])

ind = np.argmax(lons >= 180)

arr = np.roll(arr,axis = 3, shift = -ind)

print(arr)

    