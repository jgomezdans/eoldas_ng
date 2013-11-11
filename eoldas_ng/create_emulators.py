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
        print "Dumping to %s (RMSE:%g)" % ( fnames[i], rmse )
        emu.dump_emulator ( fnames[i] )
        gps.append ( emu )
        
    return gps

def do_band_emulation ( ):
    """
    TODO: not even there yet!!!
    """
    pass

def do_mv_emulation ( xtrain, xvalidate, train_brf, validate_brf ):
    N = xvalidate.shape[0]
    emu = MultivariateEmulator (y=xtrain, X=train_brf )
    rmse = np.sqrt(np.mean([(validate_brf[i] - \
         emu.predict ( np.atleast_2d(xvalidate[i,:]))[0])**2 \
         for i in xrange(N)]))
    return emu, rmse



if __name__ == "__main__":
    from collections import OrderedDict
    from operators import *
    FIXED = 1
    CONSTANT = 2
    VARIABLE = 3
    # Create the state
    # First, define the state configuration dictionary
    
    state_config = OrderedDict ()
    
    state_config['n'] = CONSTANT
    state_config['cab'] = VARIABLE
    state_config['car'] = CONSTANT
    state_config['cbrown'] = VARIABLE
    state_config['cw'] = VARIABLE
    state_config['cm'] = VARIABLE
    state_config['lai'] = VARIABLE
    state_config['ala'] = CONSTANT
    state_config['lidfb'] = FIXED
    state_config['bsoil'] = CONSTANT
    state_config['psoil'] = VARIABLE
    state_config['hspot'] = CONSTANT
    
    
    
    
    # Now define the default values
    default_par = OrderedDict ()
    default_par['n'] = 1.5
    default_par['cab'] = 40.
    default_par['car'] = 10.
    default_par['cbrown'] = 0.01
    default_par['cw'] = 0.018 # Say?
    default_par['cm'] = 0.0065 # Say?
    default_par['lai'] = 2
    default_par['ala'] = 45.
    default_par['lidfb'] = 0.
    default_par['bsoil'] = 1.
    default_par['psoil'] = 0.1
    default_par['hspot'] = 0.01
    
    
    # Define boundaries
    #parameter_names = [ 'bsoil', 'cbrown', 'hspot', 'n', \
    #    'psoil', 'ala', 'lidfb', 'cab', 'car', 'cm', 'cw', 'lai' ]
    parameter_min = OrderedDict()
    parameter_max = OrderedDict()
    
    min_vals = [ 0.8, 0.2, 0.0, 0.0, 0.0043, 0.0017, 0.001, 0., 0., 0., 0., 0.001]
    max_vals = [2.5, 77., 25., 1., 0.0713, 0.0331, 8., 90., 0., 2., 8., 0.999]


        
        
    for i, param in enumerate ( state_config.keys() ):
        parameter_min[param] = min_vals[i]
        parameter_max[param] = max_vals[i]
    # Define the state grid. In time in this case
    state_grid = np.arange ( 1, 366 )
    # Define parameter transformations
    transformations = {
        'lai': lambda x: np.exp ( -x/2. ), \
        'cab': lambda x: np.exp ( -x/100. ), \
        'car': lambda x: np.exp ( -x/100. ), \
        'cw': lambda x: np.exp ( -50.*x ), \
        'cm': lambda x: np.exp ( -100.*x ), \
        'ala': lambda x: x/90. }
    inv_transformations = {
        'lai': lambda x: -2*np.log ( x ), \
        'cab': lambda x: -100*np.log ( x ), \
        'car': lambda x: -100*np.log( x ), \
        'cw': lambda x: (-1/50.)*np.log ( x ), \
        'cm': lambda x: (-1/100.)*np.log ( x ), \
        'ala': lambda x: 90.*x }
    
    # Define the state
    # L'etat, c'est moi
    state = State ( state_config, state_grid, default_par, \
        parameter_min, parameter_max )
    # Set the transformations
    state.set_transformations ( transformations, inv_transformations )
    vza = [30.]
    sza = [0.]
    raa = [40.]
    fnames = [ "/tmp/vza_30_sza_0_raa_40" ]
    gps = create_emulators ( state, fnames, angles= [[30, 0, 40]])