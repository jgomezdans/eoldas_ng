#!/usr/bin/env python
"""
eoldas_ng helper functions/classes
=======================================

Here we provide  a set of helper functions/classes that provide some often
required functionality. None of this is *actually* required to run the system,
but it should make the library more user friendly.

``StandardStatePROSAIL`` and ``StandardStateSEMIDISCRETE``
------------------------------------------------------------

Potentially, there will be other classes like these two. These two classes
simplify the usage of two standard RT models in defining the state. This means
that they'll provide parameter boundaries and standard transformations.


"""

__author__  = "J Gomez-Dans"
__version__ = "1.0 (29.12.2014)"
__email__   = "j.gomez-dans@ucl.ac.uk"

from collections import OrderedDict
from operators import *
from state import State

class StandardStatePROSAIL ( State ):
    """A standard state configuration for the PROSAIL model"""
    def __init__ ( self, state_config, state_grid, \
                 output_name=False, verbose=False ):
        
        self.state_config = state_config
        self.state_grid = state_grid
        self.n_elems =  self.state_grid.size
        # Now define the default values
        default_par = OrderedDict ()
        default_par['n'] = 2.
        default_par['cab'] = 20.
        default_par['car'] = 1.
        default_par['cbrown'] = 0.01
        default_par['cw'] = 0.018 # Say?
        default_par['cm'] = 0.0065 # Say?
        default_par['lai'] = 2
        default_par['ala'] = 70.
        default_par['bsoil'] = 0.5
        default_par['psoil'] = 0.9
        
        self.operators = {}
        self.n_params = self._state_vector_size ()
        self.parameter_min = OrderedDict()
        self.parameter_max = OrderedDict()
        min_vals = [ 0.8, 0.2, 0.0, 0.0, 0.0043, 0.0017,0.01, 0, 0., -1., -1.]
        max_vals = [2.5, 77., 5., 1., 0.0753, 0.0331, 8., 90., 2., 1.]

        for i, param in enumerate ( state_config.keys() ):
            self.parameter_min[param] = min_vals[i]
            self.parameter_max[param] = max_vals[i]

        self.verbose = verbose
        self.bounds = []
        for ( i, param ) in enumerate ( self.state_config.iterkeys() ):
            self.bounds.append ( [ self.parameter_min[param]*0.9, \
                self.parameter_max[param]*1.1 ] )
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

        
        self.set_transformations ( transformations, inv_transformations )
        if output_name is None:
            tag = time.strftime( "%04Y%02m%02d_%02H%02M%02S_", time.localtime())
            tag += platform.node()
            self.output_name = "eoldas_retval_%s.pkl" % tag
            
        else:
            self.output_name = output_name
        print "Saving results to %s" % self.output_name

class StandardStateSEMIDISCRETE ( State ):
    """A standard state configuration for the SEMIDISCRETE model"""
    def __init__ ( self, state_config, state_grid, \
                 output_name=False, verbose=False ):
        
        self.state_config = state_config
        self.state_grid = state_grid
        self.n_elems =  self.state_grid.size
        
        self.default_values = default_values
        self.operators = {}
        self.n_params = self._state_vector_size ()
        self.parameter_min = parameter_min
        self.parameter_max = parameter_max
        self.verbose = verbose
        self.bounds = []
        for ( i, param ) in enumerate ( self.state_config.iterkeys() ):
            self.bounds.append ( [ self.parameter_min[param]*0.9, \
                self.parameter_max[param]*1.1 ] )
        self.invtransformation_dict = {}
        self.transformation_dict = {}
        self.set_transformations ( transformations_dict, invtransformation_dict )
        if output_name is None:
            tag = time.strftime( "%04Y%02m%02d_%02H%02M%02S_", time.localtime())
            tag += platform.node()
            self.output_name = "eoldas_retval_%s.pkl" % tag
            
        else:
            self.output_name = output_name
        print "Saving results to %s" % self.output_name


class CrossValidation ( object ):
    """A crossvalidation class for eoldas_ng"""
    pass
