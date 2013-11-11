#!/usr/bin/env python
import sys
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



if __name__ == '__main__':
    import random

    # Make and use a persistent dictionary
    with PersistentDict('/tmp/demo.json', 'c', format='json') as d:
        print(d, 'start')
        d['abc'] = '123'
        d['rand'] = random.randrange(10000)
        print(d, 'updated')

    # Show what the file looks like on disk
    with open('/tmp/demo.json', 'rb') as f:
        print(f.read())



def create_emulators ( state, v_size=200, n_size=100, vzax = np.arange ( 5, 60, 5 ), \
       szax = np.arange ( 5, 60, 5 ), raax = np.arange (-180, 180, 30 )  ):

    distributions = []
    for param, minval in state.parameter_min.iteritems():
        if param != "lidfb":
            maxval = state.parameter_max[param]
            distributions.append ( ss.uniform( loc=minval, scale=(maxval-minval)) )
    samples = lhd ( dist=distributions, size=n_size )
    # We need to add the lidfb parameter to the array
    x = np.zeros((100, 12)
    x[:, :9] = samples[:,:9]
    x[:, -2:] = samples[:, -2:]
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
    S[:, :12] = X
    S[:, -1] = 2

    for angle in angles:
        S[:, 12:15] = angle 
        V[:, 12:15] = angle
        train_brf = np.array ( [ prosail.run_prosail ( *s ) for s in S ] )
        validate_brf = np.array ( [ prosail.run_prosail ( *s ) for s in V ] )