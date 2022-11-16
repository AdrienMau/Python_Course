# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:14:31 2020

Fitting of Exponential
    y = A*Exp( -t/T ) + offset
     
    parameter p is [A, T, offset]



@author: Adrien Mau - ISMO/Abbelight

"""


import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize

""" FIT OF ONE GAUSSIAN 1D :"""
# p is [Sum (amp), x, sx, offset]

def dec_exp(p,X,fixed_offset=None):
    y = p[0]*np.exp(-X / p[1] )
    if fixed_offset is None:
        return p[-1] + y
    else:
        return fixed_offset + y

def dec_exp_diff(p,X,Y,fixed_offset=None):
    if not(fixed_offset is None):
        p = np.concatenate((p,fixed_offset))
    return dec_exp(p,X) - Y

def dec_exp_guess(Y, X=None, fixed_offset=None):
    """try to guess parameter of Y = A*Exp( -X/T ) + offset """
    #we take min as offset, max near X=0 as amplitude, and a median of the correct tho for Y = A*Exp( -X/T ) + offset
    Y[Y<0] = 0
    if not (fixed_offset is None):
        offset = fixed_offset
    else:
        offset = np.min(Y)
    if X is None:
        X = np.arange(0,len(Y) )
        
    amp = np.max( Y[ np.argmin( X ) ] ) #works if X close to 0 in the data... 
    exp_part = np.log( amp / ( Y-offset) )
    tho_at_each_X = X /np.log( exp_part )
    tho = np.nanmedian( tho_at_each_X )
        
    return [amp,tho,offset]
    
def dec_exp_fit(Y, X=None, p0=None, plotp0=False, fixed_offset=None):
    if p0 is None:
        p0 = dec_exp_guess(Y,X=X)# p is [Sum (amp), x, sx, offset]
        if not(fixed_offset is None):
            p0 = p0[:-1]
        if plotp0:
            print('Using initial parameter:'+str(p0))
            
    if X is None:
        X = np.arange(0,len(Y))
    p = optimize.leastsq(dec_exp_diff, p0 , args=(X,Y,fixed_offset), xtol = 1e-4)
    return p[0] #modified  14 Ap.2020


def dec_exp_test():
    global X,Y,Ynoise
    X = np.arange(0,500)
    pini = [500, 34.5, 50]
    Y = dec_exp(pini,X)
    
    plt.plot(X,Y,label='raw')
    
    pfit = dec_exp_fit(Y=Y,X=X, plotp0=True)
    print('\nTesting Exp fit without noise')
    print('Real p: '+str(pini))
    print('Found p: ' + str(pfit))
    
    
    Ynoise  = np.random.poisson( Y )
    
    print('\n')
    plt.plot(X,Ynoise, label='noised')
    
    pfit = dec_exp_fit(Y=Ynoise,X=X, plotp0=True)
    print('\nTesting Exp fit with Poisson noise')
    print('Real p: '+str(pini))
    print('Found p: ' + str(pfit))

    
    return pfit




    
if __name__ == "__main__":
    pfit = dec_exp_test()
    
    print('Done.')

    
    