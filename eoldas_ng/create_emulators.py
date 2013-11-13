#!/usr/bin/env python
import pickle
import json
import csv
import os
import shutil
import itertools


import numpy as np 
import scipy.stats as ss

import prosail
from gp_emulator import MultivariateEmulator, lhd


def fixnan(x):
  '''
  the RT model sometimes fails so we interpolate over nan

  This method replaces the nans in the vector x by their interpolated values
  '''
  for i in xrange(x.shape[0]):
    sample = x[i]
    ww = np.where(np.isnan(sample))
    nww = np.where(~np.isnan(sample))
    sample[ww] = np.interp(ww[0],nww[0],sample[nww])
    x[i] = sample
  return x

def create_emulators ( state, fnames, random=True, v_size=300, n_size=250, angles=None, \
       vzax = np.arange ( 5, 60, 5 ), szax = np.arange ( 5, 60, 5 ), \
       raax = np.arange (-180, 180, 30 )  ):
    """
    This function produces the sampling of parameter space based on the
    state object that it receives. It then goes to generate the forward
    models required, as well as a random sampling to validate against.
    
    TODO completeme!!!!!    
    """
    distributions = []
    for i, (param, minval ) in enumerate ( state.parameter_min.iteritems() ):
        maxval = state.bounds[i][1]
        minval = state.bounds[i][0]
        distributions.append ( ss.uniform( loc=minval, \
                scale=(maxval-minval)) )
    if random:
        samples = []
        for d in distributions:
            samples.append ( d.rvs( n_size ) )
        samples = np.array ( samples ).T
    else:
        samples = lhd ( dist=distributions, size=n_size )
    if angles is None:
        angles = np.array ( [angle \
            for angle in itertools.product ( szax, vzax, raax )] )
    validate = []
    for d in distributions:
        validate.append ( d.rvs( v_size ) )
    validate = np.array ( validate ).T
    V = np.zeros ((v_size, 16))
    V[:, :8] = validate[:, :8]
    V[:, 9:11] = validate[:, 8:]
    V[:, 11] = 0.01
    V[:, -1] = 2
    S = np.zeros ((n_size, 16))
    S[:, :8] = samples[:, :8]
    S[:, 9:11] = samples[:, 8:]
    S[:, 11] = 0.01 # hotspot
    S[:, -1] = 2
    #import pdb; pdb.set_trace()
    #tsamples =  np.zeros_like ( samples )
    #tvalidate = np.zeros_like ( validate )
    #validate1 = dict( zip(state.parameter_min.keys(), validate.tolist()))
    #samples1 = dict( zip(state.parameter_min.keys(), samples.tolist()))
    
    #for i,p in enumerate ( state.parameter_min.keys() ):
        #if state.transformation_dict.has_key(p):
            #tvalidate[i, :] = \
                #state.transformation_dict[p] ( np.array(validate1[p]))
            #tsamples[i, :] = \
                #state.transformation_dict[p] ( np.array(samples1[p]))
        #else:
            #tvalidate[i, :] = np.array(validate1[p] )
            #tsamples[i, :] = np.array(samples1[p] )

    gps = []
    for i, angle in enumerate ( angles ):
        S[:, 12:15] = angle 
        V[:, 12:15] = angle
        
        train_brf = np.array ( [ prosail.run_prosail ( *s ) for s in S ] )
        validate_brf = np.array ( [ prosail.run_prosail ( *s ) for s in V ] )
        train_brf = fixnan ( train_brf )
        validate_brf = fixnan ( validate_brf )
        emu, rmse = do_mv_emulation ( samples, validate, train_brf, validate_brf )
        print "(RMSE:%g)" % ( rmse )
        #emu.dump_emulator ( fnames[i] )
        gps.append ( emu )
        
    return gps


def do_mv_emulation ( xtrain, xvalidate, train_brf, validate_brf ):
    N = xvalidate.shape[0]
    emu = MultivariateEmulator (y=xtrain, X=train_brf )
    rmse = np.sqrt(np.mean([(validate_brf[i] - \
         emu.predict ( np.atleast_2d(xvalidate[i,:]))[0])**2 \
         for i in xrange(N)]))
    return emu, rmse


def create_parameter_trajectories ( state ):
    """This function creates the parameter trajectories as in the 
    RSE 2012 paper, just for testing"""
    t = np.arange ( 1, 366 )/365.
    parameter_grid = np.zeros ( (len(state.default_values.keys()), \
         t.shape[0] ) )
    for i, (parameter, default) in \
        enumerate ( state.default_values.iteritems() ):
            if parameter == "lai":
                parameter_grid[i,:]= 0.21 + 3.51 * (np.sin(np.pi*t)**5)
            elif parameter == "cab":
                w = np.where(t<=0.5)[0]
                parameter_grid[i,w] = 10.5 + 208.7*t[w]
                w = np.where(t>0.5)[0]
                parameter_grid[i,w] = 219.2 - 208.7*t[w]
            elif parameter == "cw":
                parameter_grid[i,:] =  0.068/5 + 0.01*np.sin(np.pi * t+0.1) *  \
                 np.sin(6*np.pi*t + 0.1)
            elif parameter == "xs1":
                parameter_grid[i,:] =2.5*(0.2 + 0.18*np.sin(np.pi*t) * \
                  np.sin(6*np.pi*t))
            else:
                parameter_grid[i,:] = default
             
    return parameter_grid
              
def create_observations ( state, parameter_grid, latitude, longitude, \
        b_min = np.array( [ 620., 841, 459, 545, 1230, 1628, 2105] ), \
        b_max = np.array( [ 670., 876, 479, 565, 1250, 1652, 2155] ) ):
    """This function creates the observations for  a given temporal evolution
    of parameters, loation, and bands. By default, we take only MODIS bands. 
    The function does a number of other things:
    1.- Calculate missing observations due to simulated cloud
    2.- Add noise
    TODO: There's a problem  with pyephem, gives silly solar altitudes!!!"""
    wv = np.arange ( 400, 2501 )
    band_pass = np.zeros((7,2101), dtype=np.bool)
    n_bands = b_min.shape[0]
    bw = np.zeros( n_bands )
    bh = np.zeros( n_bands )
    for i in xrange( n_bands ):
        band_pass[i,:] = np.logical_and ( wv >= b_min[i], \
                wv <= b_max[i] )
        bw[i] = b_max[i] - b_min[i]
        bh[i] = ( b_max[i] + b_min[i] )/2.
    import ephem
    o = ephem.Observer()
    o.lat, o.long, o.date = latitude, longitude, "2011/1/1 10:30"
    dd = o.date

    every = 7
    t = np.arange ( 1, 366 )
    obs_doys = np.array ( [ i for i in t if i % every == 0 ] )
    

    prop = 0.7
    WINDOW = 3
    weightings = np.repeat(1.0, WINDOW) / WINDOW
    
    xx = np.convolve(np.random.rand(len(t)*100),weightings,'valid')[WINDOW:WINDOW+len(t)]

    maxx = sorted(xx)[:int(len(xx)*prop)]
    mask = np.in1d(xx,maxx)
    doys_nocloud = t[mask]
    x = np.in1d ( obs_doys, doys_nocloud )
    obs_doys = obs_doys[x]
    vza = np.zeros_like ( obs_doys )
    sza = np.zeros_like ( obs_doys )
    raa = np.zeros_like ( obs_doys )
    rho = np.zeros (( n_bands, obs_doys.shape[0] ))
    sigma_obs = (0.01-0.004)*(bh-bh.min())/(bh.max()-bh.min())
    sigma_obs += 0.004
    for i,doy in enumerate(obs_doys):
        j = doy - 1 # location in parameter_grid...
        vza[i] = 15.#np.random.rand(1)*15. # 15 degs 
        o.date = dd + doy
        sun = ephem.Sun ( o )
        sza[i] = 15.#np.random.rand(1)*35#90. - float(sun.alt )*180./np.pi
        vaa = np.random.rand(1)*360.
        saa = np.random.rand(1)*360.
        raa[i] = 0.0#vaa - saa
        p = np.r_[parameter_grid[:8,j],0,parameter_grid[8:, j], 0.01,sza[i], vza[i], raa[i], 2 ]
        r =  fixnan( np.atleast_2d ( prosail.run_prosail ( *p )) ).squeeze()
        rho[:, i] = np.array ( [r[ band_pass[ii,:]].sum()/bw[ii] \
            for ii in xrange(n_bands) ] )
        rho[:, i] += np.random.randn ( n_bands )*sigma_obs
    return obs_doys, vza, sza, raa, rho
        

