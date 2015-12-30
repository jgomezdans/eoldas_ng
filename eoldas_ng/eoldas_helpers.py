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

``CrossValidation``
---------------------

A simple cross-validation framework for models that have a ``gamma`` term. Done
by 

``SequentialVariational``
---------------------------

The world-famous sequential variational approach to big state space assimilation.


"""

__author__  = "J Gomez-Dans"
__version__ = "1.0 (29.12.2014)"
__email__   = "j.gomez-dans@ucl.ac.uk"

import platform
import numpy as np
import time

from collections import OrderedDict
from state import State, MetaState

class StandardStatePROSAIL ( State ):
    """A standard state configuration for the PROSAIL model"""
    def __init__ ( self, state_config, state_grid, \
                 optimisation_options=None, \
                 output_name=None, verbose=False ):
        
        self.state_config = state_config
        self.state_grid = state_grid
        self.n_elems =  self.state_grid.size
        # Now define the default values
        self.default_values = OrderedDict ()
        self.default_values['n'] = 1.6
        self.default_values['cab'] = 20.
        self.default_values['car'] = 1.
        self.default_values['cbrown'] = 0.01
        self.default_values['cw'] = 0.018 # Say?
        self.default_values['cm'] = 0.03 # Say?
        self.default_values['lai'] = 2
        self.default_values['ala'] = 70.
        self.default_values['bsoil'] = 0.5
        self.default_values['psoil'] = 0.9
        
        self.metadata = MetaState()
        self.metadata.add_variable ( "n","None", "PROSPECT leaf layers", "leaf_layers" )
        self.metadata.add_variable ( "cab","microgram per centimetre^2", 
                                "PROSPECT leaf chlorophyll content",
                                "cab" )
        self.metadata.add_variable ( "car",
                                "microgram per centimetre^2", 
                                "PROSPECT leaf carotenoid content",
                                "car" )
        self.metadata.add_variable ( "cbrown","fraction", 
                                    "PROSPECT leaf senescent fraction",
                                    "cbrown" )
        self.metadata.add_variable ( "cw","centimetre", 
                                    "PROSPECT equivalent leaf water",
                                    "cw" )
        self.metadata.add_variable ( "cm",
                                    "gram per centimeter^2", "PROSPECT leaf dry matter",
                                    "cm" )
        self.metadata.add_variable ( "lai","meter^2 per meter^2", "Leaf Area Index",
                                    "lai" )
        self.metadata.add_variable ( "ala","degree", "Average leaf angle",
                                    "ala" )
        self.metadata.add_variable ( "bsoil","", "Soil brightness",
                                    "bsoil" )
        self.metadata.add_variable ( "psoil","", "Soil moisture term",
                                    "psoil" )
        
        self.operators = {}
        self.n_params = self._state_vector_size ()
        self.parameter_min = OrderedDict()
        self.parameter_max = OrderedDict()
        min_vals = [ 0.8, 0.2, 0.0, 0.0, 0.0043, 0.0017,0.0001, 0, 0., -1., -1.]
        max_vals = [2.5, 77., 5., 1., 0.0753, 0.0331, 10., 90., 2., 1.]

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

        
        self._set_optimisation_options ( optimisation_options )
        self._create_output_file ( output_name )
        
        
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
    def __init__ ( self, state ):
        """The crossvalidation only requires the state object. This will have a
        dictionary that stores the different elements of the cost function.
        """
        self.state = state
        # In the next loop, we iterate over the defined operators, and store the
        # regularisation operator, as well as those having observations.
        self.observation_operators = []
        for op in state.operatrors.iterkeys():
            if getattr ( state.operators[op], 'gamma' ):
                self.regularisation_operator = op
            elif getattr ( state.operators[op], 'observations' ):
                self.observation_operators.append ( op )
    
    def _create_xval_mask ( self, fraction ):
        self.original_mask = []
        for op in self.observation_operators:
            # This is for a 2D mask... 1D time masks should be similar
            # We start by storing the original mask. Remember to copy...
            self.original_mask[op] = self.state.operators[op].mask*1. # Copy the mask
            # For 2D masks, we look for the good pixel locations
            nnz = np.nonzero ( self.original_mask[op] )
            # This is the number of good pixels
            N = len ( nnz[0] )
            # We select a random fraction of these pixels
            passers = np.random.choice ( np.arange( N ), N*fraction )
            # In the original object, we flip the Trues to False...
            self.state.operators[op].mask[nnz[0][passers], nnz[1][passers]] = False
            

    def do_crossval ( gamma_range = None ):
        """Run the crossvalidation. If you don't specify a range of gammas, we'll
        set up one by default. You've been warned...."""
        gamma_range = np.logspace ( -3, 8, 10 )
        for g in gamma_range:
            self.state.operatrors[self.regularisation_operator].gamma = g
            retval = self.state.solve () # probably needs a starting point?
            # In here, we should ensure that we forward model the entire dataset
            # Calculate some sort of mismatch metric...
        
