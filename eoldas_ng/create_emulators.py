#!/usr/bin/env python
import sys
sys.path.append ("/home/ucfajlg/Data/python/gp_emulator")
import pickle
import json
import csv
import os
import shutil
import itertools


import numpy as np 
import scipy.stats as ss

from lhd import lhd
import prosail
from multivariate_gp import MultivariateEmulator


class PersistentDict(dict):
    ''' Persistent dictionary with an API compatible with shelve and anydbm.

    The dict is kept in memory, so the dictionary operations run as fast as
    a regular dictionary.

    Write to disk is delayed until close or sync (similar to gdbm's fast mode).

    Input file format is automatically discovered.
    Output file format is selectable between pickle, json, and csv.
    All three serialization formats are backed by fast C implementations.

    '''

    def __init__(self, filename, flag='c', mode=None, format='pickle', *args, **kwds):
        self.flag = flag                    # r=readonly, c=create, or n=new
        self.mode = mode                    # None or an octal triple like 0644
        self.format = format                # 'csv', 'json', or 'pickle'
        self.filename = filename
        if flag != 'n' and os.access(filename, os.R_OK):
            fileobj = open(filename, 'rb' if format=='pickle' else 'r')
            with fileobj:
                self.load(fileobj)
        dict.__init__(self, *args, **kwds)

    def sync(self):
        'Write dict to disk'
        if self.flag == 'r':
            return
        filename = self.filename
        tempname = filename + '.tmp'
        fileobj = open(tempname, 'wb' if self.format=='pickle' else 'w')
        try:
            self.dump(fileobj)
        except Exception:
            os.remove(tempname)
            raise
        finally:
            fileobj.close()
        shutil.move(tempname, self.filename)    # atomic commit
        if self.mode is not None:
            os.chmod(self.filename, self.mode)

    def close(self):
        self.sync()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def dump(self, fileobj):
        if self.format == 'csv':
            csv.writer(fileobj).writerows(self.items())
        elif self.format == 'json':
            json.dump(self, fileobj, separators=(',', ':'))
        elif self.format == 'pickle':
            pickle.dump(dict(self), fileobj, 2)
        else:
            raise NotImplementedError('Unknown format: ' + repr(self.format))

    def load(self, fileobj):
        # try formats from most restrictive to least restrictive
        for loader in (pickle.load, json.load, csv.reader):
            fileobj.seek(0)
            try:
                return self.update(loader(fileobj))
            except Exception:
                pass
        raise ValueError('File not in a supported format')



#if __name__ == '__main__':
    #import random

    ## Make and use a persistent dictionary
    #with PersistentDict('/tmp/demo.json', 'c', format='json') as d:
        #print(d, 'start')
        #d['abc'] = '123'
        #d['rand'] = random.randrange(10000)
        #print(d, 'updated')

    ## Show what the file looks like on disk
    #with open('/tmp/demo.json', 'rb') as f:
        #print(f.read())

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

def create_emulators ( state, fnames, v_size=200, n_size=200, angles=None, \
       vzax = np.arange ( 5, 60, 5 ), szax = np.arange ( 5, 60, 5 ), \
       raax = np.arange (-180, 180, 30 )  ):
    """
    This function produces the sampling of parameter space based on the
    state object that it receives. It then goes to generate the forward
    models required, as well as a random sampling to validate against.
    
    TODO completeme!!!!!    
    """
    distributions = []
    for param, minval in state.parameter_min.iteritems():
        if param != "lidfb":
            maxval = state.parameter_max[param]
            distributions.append ( ss.uniform( loc=minval, \
                scale=(maxval-minval)) )
    samples = lhd ( dist=distributions, size=n_size )
    # We need to add the lidfb parameter to the array
    x = np.zeros(( n_size, 11))
    x[:, :9] = samples[:,:9]
    x[:, -2:] = samples[:, -2:]
    if angles is None:
        angles = np.array ( [angle \
            for angle in itertools.product ( szax, vzax, raax )] )
    validate = []
    for d in distributions:
        validate.append ( d.rvs( v_size ) )
    validate = np.array ( validate ).T
    V = np.zeros ((v_size, 16))
    V[:, :9] = validate[:, :9]
    V[:, 10:12] = validate[:, -2:]
    V[:, -1] = 2
    S = np.zeros ((n_size, 16))
    S[:, :9] = samples[:, :9]
    S[:, 10:12] = samples[:, -2:]
    S[:, -1] = 2

    gps = []
    for i, angle in enumerate ( angles ):
        S[:, 12:15] = angle 
        V[:, 12:15] = angle
        train_brf = np.array ( [ prosail.run_prosail ( *s ) for s in S ] )
        validate_brf = np.array ( [ prosail.run_prosail ( *s ) for s in V ] )
        train_brf = fixnan ( train_brf )
        validate_brf = fixnan ( validate_brf )
        emu, rmse = do_mv_emulation ( x, validate, train_brf, validate_brf )
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
        vza[i] = np.random.rand(1)*15. # 15 degs 
        o.date = dd + doy
        sun = ephem.Sun ( o )
        sza[i] = np.random.rand(1)*35#90. - float(sun.alt )*180./np.pi
        vaa = np.random.rand(1)*360.
        saa = np.random.rand(1)*360.
        raa[i] = vaa - saa
        p = np.r_[parameter_grid[:, j], sza[i], vza[i], raa[i], 2 ]
        r =  fixnan( np.atleast_2d ( prosail.run_prosail ( *p )) ).squeeze()
        rho[:, i] = np.array ( [r[ band_pass[ii,:]].sum()/bw[ii] \
            for ii in xrange(n_bands) ] )
        rho[:, i] += np.random.randn ( n_bands )*sigma_obs
    return obs_doys, vza, sza, raa, rho
        

