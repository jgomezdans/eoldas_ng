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

class StandardStatePROSAIL ( State ):
    """A standard state configuration for the PROSAIL model"""
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

