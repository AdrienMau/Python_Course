# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:57:21 2022

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
from matplotlib.widgets import Slider

import tifffile as tf
import os, shutil
plt.style.use('default')
plt.rcParams['image.cmap'] = 'plasma' 

import scipy
from skimage import filters
from skimage import measure
# import imageio

import Exp

def imshow_slider( img , aspect='equal' , my_cmap='viridis', update_cbar=True ):
    """ imshow a 3D image with z slider option for first axis.
    update_cbar : vmin and vmax values of colorbar changes.. """
    

    # The function to be called anytime a slider's value changes (boundaries change)
    def update_image(zpos):  
        zpos = int(zpos)
        img_toshow = img[zpos]

        vm = img_toshow.max() 
        im_handler.set_array( img_toshow )
        if update_cbar:
            im_handler.set_clim(vmin= 0, vmax= vm )
            cbar.vmax = vm
            cbar.vmin = 0
        #todo : change colorbar scale...

    nz = img.shape[0]-1
    zpos = 0
        
    im_handlers = []
    lu_fig = plt.subplots( 1,1 , figsize=(5,5))

    if update_cbar:
        vm = img[zpos].max() #will be updated for each img zpos
    else:
        vm = img.max() #will never be updated, we take high value to begin with and stay with it.

    im_handler = plt.imshow( img[zpos], aspect=aspect, cmap=my_cmap , 
                            vmin = 0 ,
                            vmax = vm )
    cbar = plt.colorbar(fraction=0.046, pad=0.04);plt.xticks([]);plt.yticks([])  
    im_handlers.append( im_handler )

    # Make a horizontal slider to control the z
    axz = plt.axes([0.2, 0.05, 0.65, 0.02], facecolor='lightgoldenrodyellow')
    my_sliderz = Slider(
        ax=axz,         label='Z', 
        valmin=0, valmax=nz, valinit=0, valstep=1 )
    my_sliderz.on_changed( update_image )

    return my_sliderz


# =============================================================================
# 


# =============================================================================
# Control Board
# =============================================================================
file =  "//134.157.171.100/shared/adrien/Soledad Ca2+ Data/03-spon-1,25x01.tif"

file = "C:/Users/Administrateur/Documents/Python/Cours/03-spon-1,25x01.tif"

if not(os.path.isfile(file)):
    print('Incorrect file path !')

analyze_temporal_trace = True


label_signal_regions = 1
apply_gfilter = 1
do_binary_opening = True
apply_gfilter_on_bin = False
get_regions = True

# =============================================================================
# Opening Data : 
# =============================================================================

#spatial cropping of image:
crop_x = [ 100,-100]
crop_y = [ 100,-100] 
crop_z = [1000,2000]


if not 'img' in locals():    
    print('Opening image')
    img = tf.imread( file )
    print('Done')
    print('Cropping')
    img = img[ crop_z[0]:crop_z[1] , crop_y[0]:crop_y[1], crop_x[0]:crop_x[1] ]
    print('Final img size: ', img.shape )
    
    img = img - np.min( img ) 


print( img.shape )

print('Removing median background')
median_img = np.median(img,axis=0).astype('uint16')
img_cor = (img.astype('int') - median_img) #note: img is now int32
img_cor[ img_cor<1 ] = 0
img_cor = img_cor.astype('uint16')


a = imshow_slider( img_cor , my_cmap ='nipy_spectral')


if analyze_temporal_trace:
    print('Analyzing temporal trace')
    plt.figure( figsize=(6,3))
    plt.title('Global intensity along time')
    plt.plot( np.mean(img_cor, axis=(1,2) ) )
    plt.xlabel('time')
    plt.ylabel('mean Intensity')
    plt.tight_layout()
    
    
    zm,ym,xm = np.unravel_index( np.argmax( img_cor ) , img_cor.shape )
    plt.figure()
    plt.title('highest intensity z image')
    plt.imshow( img_cor[zm] )
    plt.scatter( xm,ym ) 
    
    #note: here we follow the time trace of a single point (trace = img_cor[ :, ym,xm])
    #best would be to integrate on an area (on all the concerned area would be perfect)
    t = np.arange(0,img_cor.shape[0])
    trace_pt = img_cor[ :, ym,xm]
    trace = np.mean( img_cor[ :, ym-5:ym+6,xm-5:xm+6] ,axis=(1,2) ) #sum on a small area.
    
    plt.figure()
    plt.plot( t,trace , label='mean trace')
    plt.plot( t,trace_pt , color='k', alpha=0.5, label='point trace')
    plt.xlabel('time')
    plt.ylabel('Intensity')
    
    
    thres_trace = 2
    print('Threshold for trace is ', thres_trace )
    
    t_signal = trace>thres_trace #time pos where there is significant signal.
    plt.plot( t, np.repeat( thres_trace, len(t)) , linestyle=':' )

    #analyzing a single trace : 
    single_peak = trace[295:350]
    single_t = t[295:350]
    single_t = single_t - np.min(single_t) 
    
    pinit = Exp.dec_exp_guess( X=single_t, Y=single_peak  )
    p = Exp.dec_exp_fit( X=single_t, Y=single_peak , p0 = pinit )
    efit =  Exp.dec_exp( p, single_t )
    tau = round(p[1],2) #tho in pixel.
    
    plt.figure('trace')
    plt.plot( single_t ,single_peak, label='trace' , color='k')
    plt.plot( single_t ,efit, label=r'fit, $\tau=$'+str(tau), linestyle=':' , color='indigo')
    plt.legend()
    

    
    


    
    
if label_signal_regions:
    
    if apply_gfilter:
        print('\t Applying G filter on image (long).')
        img_cor_g = filters.gaussian( img_cor, sigma=5 ) #long... img is then float...
    else:
        img_cor_g = img_cor
    thres = filters.threshold_otsu( img_cor_g )
    print('Applying a threshold of ', thres )
    
    
    img_cor_bin = img_cor_g>thres 
    
    if do_binary_opening:
        print('\t Using binary opening on imbin.')
        img_cor_bin = scipy.ndimage.binary_opening( img_cor_bin, structure=np.ones((1,3,3),dtype='bool')).astype('bool')
    
    if apply_gfilter_on_bin:
        print('\t Applying G filter on binned image (long).')
        img_cor_bin = filters.gaussian( img_cor_bin, sigma=6 ) #long... img is then float...
        
        
    if get_regions:        
        print('Estimating isolated regions')
        #tiny bit long
        labeledImage = measure.label( img_cor_bin, connectivity=3) -1 #numerote les régions indépendantes.
        
        my_cmap = plt.cm.prism.copy()
        my_cmap.set_under( color='white' ) #values <vmin (0) will be white
        b = imshow_slider( labeledImage , my_cmap = my_cmap, update_cbar=False )
    
        plt.title('found regions numbers')
        print('\t found ' + str(labeledImage.max()) + ' different regions')
        
        #measurement of volume (for example) :
        uid, volume = np.unique(labeledImage, return_counts=True)
        
        plt.figure()
        plt.hist( np.ravel(labeledImage), bins=labeledImage.max() , range=(1,labeledImage.max()))
        plt.xlabel('zone number')
        plt.ylabel('volume ($pixel^{3}$)')
        
        plt.figure()
        plt.hist( volume[1:], bins=200)
        plt.xlabel('volume')
        plt.ylabel('narea')
        
        
        # #For each region we will try to have Imean(t) :
        # #we store for each region, time and Imean :
        # region_ti = []
        # t = np.repeat( np.ones( ))
        # for i in uid:
        #     filt = (labeledImage==uid) #slow.
        #     intensity = img_cor_g[ filt ]
                
                
                
        tf.imsave( 'labeledimg.tif', labeledImage  )
            
