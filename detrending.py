# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:21:04 2020

@author: gabin
"""

import netCDF4 as nc
import numpy as np
import os.path 
import scipy.optimize as so
import matplotlib.pylab as plt 
import pandas 
import copy 
from scipy import optimize
from scipy import linalg 
from scipy import spatial
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata  
from scipy.interpolate import interp1d 
import calendar
from scipy import interpolate


###################################################
# Creation of swot projections (on obs or model)  #
###################################################              
def obs_swotprojection(ssh_swot,flag_plot=False,boxsize = 99999):
    """
    NAME 
        obs_swotprojection
    DESCRIPTION 
        Produce and return the projection T defined in Metref et al. 2019.  
        
        Args:       
            ssh_swot (float array [nacross,nalong]): SWOT data,
            flag_plot (boolean): Flag if plots are produced,
            boxsize (int): Along track size of the sliding average box where the CER is performed
  
        Returns: 
            ssh_detrended (float array [nacross,nalong]): Projection T of SWOT, 
            aa1,bb1,cc1,ee11,ee12,ff11,ff12 (float array [nalong]): Regression coefficients 
    """      
   
    swot_fill_value = 2147483647. 
        
   
    # Mask invalid values 
    ssh_swotgrid=np.ma.masked_where(ssh_swot==swot_fill_value, ssh_swot,copy=False)   
    np.ma.masked_invalid(ssh_swotgrid) 
    
    # Plot SWOT observation
    if flag_plot:
            plt.figure()
            max_range=0.3 
            plt.imshow(ssh_swotgrid)  
            plt.colorbar(extend='both', fraction=0.042, pad=0.04)  
            plt.clim(-max_range,max_range) 
            plt.title('SWOT')
            plt.show()
          
    # Swath gap size (in gridpoints)
    n_gap = 10 
    
    # Calling obs_detrendswot
    ssh_detrended, aa1,bb1,cc1,ee11,ee12,ff11,ff12 = obs_detrendswot(ssh_swotgrid,n_gap,False,boxsize=boxsize)
    
    
    # Mask invalid values  
    np.ma.masked_invalid(ssh_detrended)  
          
        
        
    if flag_plot:
            plt.figure() 
            plt.imshow(ssh_detrended)  
            plt.colorbar(extend='both', fraction=0.042, pad=0.04)   
            plt.title('SWOT CER projection')
            plt.show()
            
            plt.figure()   
            plt.imshow(ssh_detrended-ssh_swotgrid)  
            plt.colorbar(extend='both', fraction=0.042, pad=0.04)   
            plt.title('SWOT CER projection - SWOT')
            plt.show()
          
         
    return ssh_detrended, aa1,bb1,cc1,ee11,ee12,ff11,ff12 


def obs_detrendswot(ssh_swotgrid0, n_gap, removealpha0=False,boxsize = 99999):
    """
    NAME 
        obs_detrendswot
    DESCRIPTION 
        Create the projection T (defined in Metref et al. 2019) of ssh_swotgrid0, an array of SSH in SWOT grid.
        
        Args:       
            ssh_swot (float array [nacross,nalong]): SWOT data,
            ngap (int): Size of the across track gap between the two SWOT swath,
            flag_plot (boolean): Flag if plots are produced,
            boxsize (int): Along track size of the sliding average box where the CER is performed
  
        Returns: 
            ssh_detrended0 (float array [nacross,nalong]): Projection T of SWOT, 
            aa1,bb1,cc1,ee11,ee12,ff11,ff12 (array[nalong]): Regression coefficients 
    """    
    # Size of averaging box for regression coefficient (only works if meanafter == True)
    #boxsize = 500 
    # Lenght of the swath
    dimtime = np.shape(ssh_swotgrid0)[0]
    # Width of the swath
    dimnc = np.shape(ssh_swotgrid0)[1]

    ssh_detrended0 = np.zeros_like(ssh_swotgrid0)
    ssh_detrended0[:,:] = ssh_swotgrid0[:,:]
    
    # Initialization of the regression coefficients
    a1=0
    b1=0 
    c1=0 
    e11=0
    e12=0 
    f11=0
    f12=0 
    
    # Three options: 
    # - typemean == 1: computing the regression coefficients (i_along times) on the sliding mean (ssh_acrosstrack)
    # - typemean == 2: averaging the regression coefficients computed at each ssh_acrosstrack (!! a lot slower !!)
    # - typemean == 3: computing the regression coefficients only once on the overall mean (ssh_acrosstrack)
    
    typemean = 2
    
    if boxsize > dimtime: 
        typemean = 3 # True # False # 
 
    #########################################
    # Calculate the regression coefficients #
    #########################################
    if typemean == 1:
        aa1=np.zeros(dimtime) 
        bb1=np.zeros(dimtime) 
        cc1=np.zeros(dimtime) 
        ee11=np.zeros(dimtime)
        ee12=np.zeros(dimtime)
        ff11=np.zeros(dimtime)
        ff12=np.zeros(dimtime)
        i_along_init = 0
        i_along_fin = 1
        # Loop on the along track
        for i_along in range(dimtime):
            ssh_across = ssh_swotgrid0[i_along,ssh_swotgrid0[i_along,:]<999.]
            ssh_across = ssh_across[ssh_across>-999.]
            if np.shape(ssh_across)[0]==dimnc:  
                if i_along_init == 0:
                    i_along_init = i_along
                nn = np.shape(ssh_across)[0] 
                x_across = np.where(ssh_swotgrid0[i_along,:]<999.)[0]-int(np.shape(ssh_swotgrid0)[1]/2)  
                x_across[x_across<0]=x_across[x_across<0]-n_gap/2
                x_across[x_across>=0]=x_across[x_across>=0]+n_gap/2
                i_along_fin = i_along

                # Cost function 
                def linreg3(params): 
                    return np.sum( ( ssh_across-(params[0]+params[1]*x_across + params[2]*x_across**2 +np.append(params[3]+params[4]*x_across[x_across<=0],params[5]+params[6]*x_across[x_across>0],axis=0) ) )**2 ) 

                # Perform minimization 
                params = np.array([a1,b1,c1,e11,e12,f11,f12])
                coefopt = so.minimize(linreg3, params, method = "Powell") 
                aa1[i_along], bb1[i_along], cc1[i_along], ee11[i_along], ee12[i_along], ff11[i_along], ff12[i_along] = coefopt['x'][0], coefopt['x'][1], coefopt['x'][2], coefopt['x'][3], coefopt['x'][4], coefopt['x'][5], coefopt['x'][6] 
                a1=aa1[i_along]
                b1=bb1[i_along] 
                c1=cc1[i_along] 
                e11=ee11[i_along]
                e12=ee12[i_along]
                f11=ff11[i_along]
                f12=ff12[i_along]   
        if i_along_init==i_along_fin:
            i_along_fin=i_along_fin+1 
    elif typemean == 2 :  
        
        ssh_masked = np.ma.masked_where(ssh_swotgrid0>999.,ssh_swotgrid0) 
        ssh_masked = np.ma.masked_where(ssh_masked<-999.,ssh_masked) 
        aa1=np.zeros(dimtime) 
        bb1=np.zeros(dimtime) 
        cc1=np.zeros(dimtime) 
        ee11=np.zeros(dimtime)
        ee12=np.zeros(dimtime)
        ff11=np.zeros(dimtime)
        ff12=np.zeros(dimtime)
        i_along_init = 0
        i_along_fin = 1
        # Loop on the along track
        for i_along in range(dimtime): 
            # Average the ssh (sliding box)
            ssh_across = np.nanmean(ssh_masked[max(0,int(i_along-boxsize/2)):min(dimtime,int(i_along+boxsize/2)),:],0)  
            if np.shape(ssh_across)[0]==dimnc:  
                if i_along_init == 0:
                    i_along_init = i_along
                nn = np.shape(ssh_across)[0] 
                if nn != 0: 
                    x_across = np.arange(nn)-int(nn/2) 
                x_across[x_across<0]=x_across[x_across<0]-n_gap/2
                x_across[x_across>=0]=x_across[x_across>=0]+n_gap/2
                i_along_fin = i_along

                # Cost function
                def linreg3(params): 
                    return np.sum( ( ssh_across-(params[0]+params[1]*x_across + params[2]*x_across**2 +np.append(params[3]+params[4]*x_across[x_across<=0],params[5]+params[6]*x_across[x_across>0],axis=0) ) )**2 ) 

                # Perform minimization 
                params = np.array([a1,b1,c1,e11,e12,f11,f12])
                coefopt = so.minimize(linreg3, params, method = "Powell") 
                aa1[i_along], bb1[i_along], cc1[i_along], ee11[i_along], ee12[i_along], ff11[i_along], ff12[i_along] = coefopt['x'][0], coefopt['x'][1], coefopt['x'][2], coefopt['x'][3], coefopt['x'][4], coefopt['x'][5], coefopt['x'][6] 
                a1=aa1[i_along]
                b1=bb1[i_along] 
                c1=cc1[i_along] 
                e11=ee11[i_along]
                e12=ee12[i_along]
                f11=ff11[i_along]
                f12=ff12[i_along]    
    elif typemean == 3 :    
            ssh_across = np.nanmean(ssh_swotgrid0,0) 

            if np.shape(ssh_across)[0]==dimnc:  
                nn = np.shape(ssh_across)[0] 
                if nn != 0: 
                    x_across = np.arange(nn)-int(nn/2)    
                    x_across[x_across<0]=x_across[x_across<0]-n_gap/2+1 
                    x_across[x_across>=0]=x_across[x_across>=0]+n_gap/2   

                    if nn == np.shape(x_across)[0]: 
                        # Cost function
                        def linreg3(params): 
                            return np.sum( ( ssh_across-(params[0]+params[1]*x_across + params[2]*x_across**2 +np.append(params[3]+params[4]*x_across[x_across<=0],params[5]+params[6]*x_across[x_across>0],axis=0) ) )**2 ) 

                        # Perform minimization  
                        params = np.array([a1,b1,c1,e11,e12,f11,f12])
                        coefopt = so.minimize(linreg3, params, method = "Powell") 
                        a1,b1,c1,e11,e12,f11,f12 = coefopt['x'][0], coefopt['x'][1], coefopt['x'][2], coefopt['x'][3], coefopt['x'][4], coefopt['x'][5], coefopt['x'][6]  

    
    ######################################################################
    # Make projection using the regression coefficients calculated above #
    ######################################################################
    for i_along in range(dimtime):  
        # Organize the coefficients for the projection 
        ssh_across = ssh_swotgrid0[i_along,ssh_swotgrid0[i_along,:]<999.]
        if typemean == 1 :
                # Average the coefficients (sliding box)
                if int(max(i_along_init,i_along-boxsize/2)) < int(min(i_along+boxsize/2,i_along_fin)):
                    a1 = np.mean(aa1[int(max(i_along_init,i_along-boxsize/2)):int(min(i_along+boxsize/2,i_along_fin))])
                    b1 = np.mean(bb1[int(max(i_along_init,i_along-boxsize/2)):int(min(i_along+boxsize/2,i_along_fin))]) 
                    c1 = np.mean(cc1[int(max(i_along_init,i_along-boxsize/2)):int(min(i_along+boxsize/2,i_along_fin))]) 
                    e11 = np.mean(ee11[int(max(i_along_init,i_along-boxsize/2)):int(min(i_along+boxsize/2,i_along_fin))])
                    e12 = np.mean(ee12[int(max(i_along_init,i_along-boxsize/2)):int(min(i_along+boxsize/2,i_along_fin))])
                    f11 = np.mean(ff11[int(max(i_along_init,i_along-boxsize/2)):int(min(i_along+boxsize/2,i_along_fin))])
                    f12 = np.mean(ff12[int(max(i_along_init,i_along-boxsize/2)):int(min(i_along+boxsize/2,i_along_fin))]) 
                else:
                    if i_along-boxsize/2 >= i_along_fin:  
                        a1 = np.mean(aa1[int(i_along_fin)])
                        b1 = np.mean(bb1[int(i_along_fin)]) 
                        c1 = np.mean(cc1[int(i_along_fin)]) 
                        e11 = np.mean(ee11[int(i_along_fin)])
                        e12 = np.mean(ee12[int(i_along_fin)])
                        f11 = np.mean(ff11[int(i_along_fin)])
                        f12 = np.mean(ff12[int(i_along_fin)])
                    if i_along_init >= i_along+boxsize/2:  
                        a1 = np.mean(aa1[int(i_along_init)])
                        b1 = np.mean(bb1[int(i_along_init)]) 
                        c1 = np.mean(cc1[int(i_along_init)]) 
                        e11 = np.mean(ee11[int(i_along_init)])
                        e12 = np.mean(ee12[int(i_along_init)])
                        f11 = np.mean(ff11[int(i_along_init)])
                        f12 = np.mean(ff12[int(i_along_init)])
                         
        elif typemean == 2: 
                a1 = aa1[i_along]
                b1 = bb1[i_along]
                c1 = cc1[i_along]
                e11 = ee11[i_along]
                e12 = ee12[i_along]
                f11 = ff11[i_along]
                f12 = ff12[i_along]  
                
        elif typemean == 3: 
                aa1=np.zeros(dimtime) + a1 
                bb1=np.zeros(dimtime) + b1 
                cc1=np.zeros(dimtime) + c1 
                ee11=np.zeros(dimtime) + e11
                ee12=np.zeros(dimtime) + e12
                ff11=np.zeros(dimtime) + f11
                ff12=np.zeros(dimtime) + f12 
                
        # Make the projection         
        nn = np.shape(ssh_across)[0] 
        x_across = np.where(ssh_swotgrid0[i_along,:]<999.)[0]-int(np.shape(ssh_swotgrid0)[1]/2) 
        x_across[x_across<0]=x_across[x_across<0]-n_gap/2
        x_across[x_across>=0]=x_across[x_across>=0]+n_gap/2 
        x_across_sized = x_across[ssh_across<999.]  
        if removealpha0 :
                ssh_detrended0[i_along,ssh_swotgrid0[i_along,:]<999.] = ssh_across[ssh_across<999.] - (a1+b1*x_across_sized + c1*x_across_sized**2+np.append(e11+e12*x_across_sized[x_across_sized<=0],f11+f12*x_across_sized[x_across_sized>0],axis=0) ) 
        else :
                ssh_detrended0[i_along,ssh_swotgrid0[i_along,:]<999.] = ssh_across[ssh_across<999.] - (b1*x_across_sized + c1*x_across_sized**2+np.append(e11+e12*x_across_sized[x_across_sized<=0],f11+f12*x_across_sized[x_across_sized>0],axis=0) )  
            

    
    return  ssh_detrended0, aa1,bb1,cc1,ee11,ee12,ff11,ff12
