# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:54:16 2022

GOAL :

STATE : In progress
    
    
NOTES : 

VERSION : 1.0



@author: Adrien Mau  - Administrateur
Institut de la Vision
CNRS / INSERM / Sorbonne Université
adrien.mau@orange.fr
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas

import os, shutil
plt.style.use('default')
plt.rcParams['image.cmap'] = 'plasma' 


file = "Data_2857_20211209_Capi3_Ls9to11_W_VitCa.xlsx"

if not 'data' in locals():
    data = pandas.read_excel( file , sheet_name='Raw' ) #a bit slow to open ?

print('Available keys are :')
for k in data.keys():
    print(k)

dk = data.keys()


#here each data column is accessed with a specific key.
#when accessed it returns a table ( i.e index (line) and value are stored together )
#it can be converted to a np.array with .values
#we will iterate over the keys, remove zeros, and contain them in lists.
times = []
values =[]
plt.figure()
for kpos,k in enumerate( dk[::2]):
    second_key = dk[2*kpos+1]
    t = data[ k ].values # '.values' converts t in a np .array
    value = data[ second_key ].values
    filt = (t!=0)
    times.append( t[filt] )
    values.append( value[filt] )
    
    plt.plot(times[-1]  ,values[-1], label=second_key )
    
    
plt.legend( fontsize=6)