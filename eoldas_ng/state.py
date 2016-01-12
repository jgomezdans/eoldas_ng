#!/usr/bin/env python

"""
The eoldas_ng state class
"""

__author__  = "J Gomez-Dans"
__email__   = "j.gomez-dans@ucl.ac.uk"

import cPickle
import platform
import collections
import time


import numpy as np
import scipy.optimize
import scipy.sparse as sp

from eoldas_utils import *
from operators import OperatorDerDerTypeError

FIXED = 1
CONSTANT = 2
VARIABLE = 3

Variable_name = collections.namedtuple ( "variable_name", 
                                        "units long_name std_name" )
class MetaState ( object ):
    """A class to store metadata on the state, such as time, location, units....
    This is required to generate CF compliant netCDF output"""
    def __init__ ( self ):
        self.metadata = {}
        
    def add_variable ( self, varname, units, long_name, std_name ):
        self.metadata[varname] = Variable_name ( units=units, 
                        long_name=long_name, std_name=std_name )
        
class State ( object ):
    
    """A state-definition class
    
    
       In EO-LDAS, the state requires the following:
        
       1. a configuration dictionary,
       2. a state grid
       3. a dictionary with default parameter values
        
       The state grid is an array that defines the domain of the
       problem. For example, for a temporal DA problem, it will be
       a vector of however many elements are required timesteps. For
       a spatial problem, it will be a 2D array with the locations
       of the gridcells. the configuration diciontary stores whether a
       particular parameter is variable over the grid (i.e. is this
       parameter estimated for all timesteps or grid positions?),
       constant (so it is constant in time and/or space), or whether we
       just prescribe some default value."""
       
    def __init__ ( self, state_config, state_grid, default_values, \
            parameter_min, parameter_max, optimisation_options=None, \
            output_name=None, verbose=False ):
        """State constructor. The state defines the problem we will try
        to solve and as such requires quite  a large number of parameters
        
        Parameters
        -----------
        state_config: OrderedDict
            The state configuration dictionary. Each key is labeled as FIXED, 
            CONSTANT or VARIABLE, indicating that that the corresponding 
            parameter is set to the default value, a constant value over the
            entire assimilation window, or variable (e.g. inferred over the 
            selected state grid).
        state_grid: array
            The grid where the parameters will be inferred. Either a 1D or a 2D
            grid            
        default_values: OrderedDict
            Default values for the variable. Should have the same keys as 
            ``state_config`` TODO: add test for keys consistency
        parameter_min: OrderedDict
            The lower boundary for the parameters. OrderedDict with same
            keys as ``state_config``  TODO: add test for keys consistency
        parameter_max: OrdederedDict
            The upper boundary for the parameters. OrderedDict with same
            keys as ``state_config``  TODO: add test for keys consistency
        optimisation_options: dict
            Configuration options for the optimiser. These are all options
            that go into scipy.optimiser TODO: link to the docs
        output_name: str
            You can give the output a string tag, or else, we'll just use
            the timestamp.
        verbose: boolean
            Whether to be chatty or not.
        netcdf: boolean
            Whether to save the output in netCDF4 format.
        
        """
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
            self.bounds.append ( [ self.parameter_min[param], \
                self.parameter_max[param] ] )
        self.invtransformation_dict = {}
        self.transformation_dict = {}
        
        
        self._set_optimisation_options ( optimisation_options )
        self._create_output_file ( output_name )
        
        
        
    def _set_optimisation_options ( self, optimisation_options ):
        if optimisation_options is None:
            self.optimisation_options = {"factr": 1000, \
                "m":400, "pgtol":1e-12, "maxcor":200, \
                "maxiter":1500, "disp":True }
        else:
            self.optimisation_options = optimisation_options

    def _create_output_file ( self, output_name ):
        
        self.netcdf = False
        if output_name is None:
            tag = time.strftime( "%04Y%02m%02d_%02H%02M%02S_", time.localtime())
            tag += platform.node()
            self.output_name = "eoldas_retval_%s" % tag
            
        elif isinstance ( output_name, basestring ):
            self.output_name = output_name + ".pkl"
        else:
            self.output_name = output_name.fname
            self.retval_file = output_name
            self.netcdf = True
            
            
        print "Saving results to %s" % self.output_name
        
    def set_metadata ( self, metadata ):
        """This method allows one to specify time and space locations for the experiment.
        These will be saved in the solution netcdf file."""
        try:
            self.metadata = metadata
        except NameError:
            raise "No netCDF4 output!"
    
    
    def set_transformations ( self, transformation_dict, \
            invtransformation_dict ):
        """We can set transformations to the data that will be
        applied automatically when required. The aim of these
        transformations is to quasi-linearise the problem, as that helps
        with convergence and with realistic estimation of posterior
        uncertainties.
        
        Parameters
        -----------
        transformation_dict: dict
            A dictionary that for each parameter (key) has a transformation 
            function going from "real units" -> "transformed units". You only
            need to specify functions for the parameters that do require a
            transformations, the others will be assumed non-transformed.
            
        invtransformation_dict: dict
            A dictionary that for each parameter (key) has the inverse
            transformation function, going from "transformed units" ->
            "real units". You only need to specify functions for the 
            parameters that do require a transformations, the others will 
            be assumed non-transformed.
        """
        self.transformation_dict = transformation_dict
        self.invtransformation_dict = invtransformation_dict
        # Recalculate boundaries
        self.bounds = []
        for ( i, param ) in enumerate ( self.state_config.iterkeys() ):
            if transformation_dict.has_key ( param ):
                tmin = transformation_dict[param] ( self.parameter_min[param] )
                tmax = transformation_dict[param] ( self.parameter_max[param] )
            else:
                tmin = self.parameter_min[param]
                tmax = self.parameter_max[param]
            if tmin > tmax:
                self.bounds.append ([ tmax, tmin ] )
            else:
                self.bounds.append ([ tmin, tmax ] )
                
                
    def _state_vector_size ( self ):
        """Returns the size of the state vector going over the 
        state grid and state config dictionary."""
        n_params = 0
        for param, typo in self.state_config.iteritems():
            if typo == CONSTANT:
                n_params  += 1
            elif typo == VARIABLE:
                n_params  += self.n_elems
        return n_params
        
    def pack_from_dict ( self, x_dict, do_transform=False ):
        """Packs a state OrderedDict dictionary into a vector that
        the function optimisers can use. Additionally, you can do a
        transform using the defined transformation dictionaries of
        functions.
        
        x_dict: OrderedDict
            A state dictionary. The state dictionary contains the state,
            indexed by parameter name (e.g. the keys of the dictionary).
            The arrays of the individual components have the true dimensions
            of state_grid. All parameter types (FIXED, CONSTANT and 
            VARIABLE) are present in the state.
        do_transform: boolean
            Whether to invoke the forward transformation method on the parameter
            (if it exists on the trnasformation dictionary) or not.
            
        Returns
        -------
        A vector of the state that can be consumed by function minimisers.
        """
        the_vector = np.zeros ( self.n_params )
        # Now, populate said vector in the right order
        # looping over state_config *should* preserve the order
        i = 0
        for param, typo in self.state_config.iteritems():
            if typo == CONSTANT: # Constant value for all times
                if do_transform and self.transformation_dict.has_key ( param ):
                    the_vector[i] = self.transformation_dict[param] ( \
                        x_dict[param] )
                else:
                    the_vector[i] = x_dict[param]
                i = i+1        
            elif typo == VARIABLE:
                # For this particular date, the relevant parameter is at location iloc
                if do_transform and self.transformation_dict.has_key ( param ):
                    the_vector[i:(i + self.n_elems)] =  \
                        self.transformation_dict[param] ( \
                            x_dict[param].flatten() )
                else:
                    the_vector[i:(i + self.n_elems)] =  \
                        x_dict[param].flatten() 
                i += self.n_elems
        return the_vector 
    
    def _unpack_to_dict ( self, x, do_transform=False, do_invtransform=False ):
        """Unpacks an optimisation vector `x` to a working dict. The oppossite of
        ``self._pack_to_dict``, would you believe it.
        
        Parameters
        -----------
        x: array
            An array with state elements
        do_transform: boolean
            Whether to do a transform from real to transformed units for the elements
            that support that.
            
        do_invtransform: boolean
            Whether to do an inverse transform from transformed units to real units for
            elements that suppor that
            
        Returns
        --------
        x_dict: OrderedDict
            A state dictionary. The state dictionary contains the state,
            indexed by parameter name (e.g. the keys of the dictionary).
            The arrays of the individual components have the true dimensions
            of state_grid. All parameter types (FIXED, CONSTANT and 
            VARIABLE) are present in the state.
        
        """
        the_dict = collections.OrderedDict()
        i = 0
        for param, typo in self.state_config.iteritems():
            
            if typo == FIXED: # Default value for all times                                
                if self.transformation_dict.has_key ( \
                        param ):
                    the_dict[param] = self.transformation_dict[param]( \
                         self.default_values[param] )
                else:
                #elif do_transform and self.transformation_dict.has_key ( \
                        #param ):
                    
                    the_dict[param] = self.default_values[param] 
                #else:
                    #the_dict[param] = self.default_values[param]
                
            elif typo == CONSTANT: # Constant value for all times
                if do_transform and self.transformation_dict.has_key ( \
                        param ):
                    the_dict[param] = self.transformation_dict[param]( x[i] )
                elif do_invtransform and self.invtransformation_dict.has_key ( \
                        param ):
                    p1 = self.transformation_dict[param] ( self.parameter_max[param] )
                    p2 = self.transformation_dict[param] ( self.parameter_min[param] )
                    pmax = max ( p1, p2 )
                    pmin = min ( p1, p2 )
                    xx = np.clip ( x[i], p1, p2 )
                    the_dict[param] = self.invtransformation_dict[param]( x[i] )

                else:
                    the_dict[param] = x[i]
                i += 1
                
            elif typo == VARIABLE:
                if do_transform and self.transformation_dict.has_key ( \
                        param ):
                    the_dict[param] = self.transformation_dict[param] ( \
                        x[i:(i+self.n_elems )]).reshape( \
                        self.state_grid.shape )
                elif do_invtransform and self.invtransformation_dict.has_key ( \
                        param ):
                    p1 = self.transformation_dict[param] ( self.parameter_max[param] )
                    p2 = self.transformation_dict[param] ( self.parameter_min[param] )
                    pmax = max ( p1, p2 )
                    pmin = min ( p1, p2 )
                    xx = np.clip ( x[i:(i+self.n_elems )].reshape( \
                        self.state_grid.shape), p1, p2 )
                    

                    the_dict[param] = self.invtransformation_dict[param] ( xx )

                else:
                    the_dict[param] = x[i:(i+self.n_elems )].reshape( \
                        self.state_grid.shape )
                i += self.n_elems
            
        return the_dict
    
    def add_operator ( self, op_name, op ):
         """Add operators to the state class. The state class per se doesn't do much, one
         needs to add operators (or "constraints"). These operators are in effect the log
         of the Gaussian difference between the state (or a transformation of it through an
         e.g. observation operator) and other constraints, be it observations, prior values,
         model expectations... The requirements for the operators are to have a ``der_cost``
         method (that returns the cost and the associated gradient) and a ``der_der_cost``,
         that returns the Hessian associated to a particular input state dictionary.
         
         Parameters
         -----------
         op_name: str 
            A name for the operator. This is just for logging and reporting to the user.
         op: Operator class
            An operator class. Typically, provided in ``operators.py`` or derived from the options
            there, but must containt ``der_cost`` and ``der_der_cost`` methods.
         """
         the_op = getattr( op, "der_cost", None)
         if not callable(the_op):
             raise AttributeError, "%s does not have a der_cost method!" % op_name     
         self.operators[ op_name ] = op
     
    def _get_bounds_list ( self ):
        """Return a list with the parameter boundaries. This is required to set the 
        optimisation boundaries, and it returns a list in the order/format expected
        by L-BFGS"""
        the_bounds = []
        for i, ( param, typo ) in enumerate(self.state_config.iteritems()):
            if typo == CONSTANT:
                the_bounds.append ( self.bounds[i] )
            elif typo == VARIABLE:
                
                [ the_bounds.append ( self.bounds[i] ) \
                    for j in xrange ( self.n_elems )]
        return the_bounds
    
    def optimize ( self, x0=None, the_bounds=None, do_unc=False, ret_sol=True ):
        """Optimise the state starting from a first guess `x0`. Can also allow the 
        specification of parameter boundaries, and whether to calculate the 
        uncertainty or not. ``x0`` can have several different forms: it can be
        an orderedDict with a first guess at the parameters, it can be an operator
        name that has a ``first_guess`` method that returns a parameter vector (this
        method is for example a way to use the inverse emulators in some cases), or
        it can be ``None``, in which case, a random state vector is used.
        
        Parameters
        -----------
        x0: dict, string or None
            Starting point for the state optimisation. Can be a state dict, a string
            indicating an operator with a ``first_guess`` method, or ``None``, which
            means that a random initialisation point will be provided.
        the_bounds: list
            Boundaries
        do_unc: boolean
            Whether to calculate the uncertainty or not.
        
        """
        
        start_time = time.clock()
        if the_bounds is None:
            the_bounds = self._get_bounds_list()        
        if (type(x0) == type ( {} ) ) or ( type(x0) == type ( collections.OrderedDict() ) ):
            # We get a starting dictionary, just use that
            x0 = self.pack_from_dict ( x0, do_transform=True )
        elif x0 is None:
            # No starting point, start from random location?
            x0 = np.array ( [ lb + (ub-lb)*np.random.rand() \
                for (lb,ub) in the_bounds ] )
            raise NotImplementedError
        elif type( x0 ) is str:
            # Use a single operator that has a ``first_guess`` method
            x0 = self.operators[x0].first_guess( self.state_config, self.state_grid.size )
            
        r = scipy.optimize.minimize ( self.cost, x0, method="L-BFGS-B", \
                jac=True, bounds=the_bounds, options=self.optimisation_options)
        end_time = time.time()
        if self.verbose:
            if r.success:
                print "Minimisation was successful: %d \n%s" % \
                        ( r.status, r.message )
            else:
                print "Minimisation was NOT successful: %d \n%s" % \
                        ( r.status, r.message )
                print "Number of iterations: %d" % r.nit
                print "Number of function evaluations: %d " % r.nfev
                print "Value of the function @ minimum: %e" % r.fun
                print "Total optimisation time: %.2f (sec)" % ( time.time() - start_time )


        #####retval['post_cov'] = post_cov
        #####retval['real_ci5pc'] = ci_5
        #####retval['real_ci95pc'] = ci_95
        #####retval['real_ci25pc'] = ci_25
        #####retval['real_ci75pc'] = ci_75
        #####retval['post_sigma'] = post_sigma
        #####retval['hessian'] =  the_hessian
                
        if self.netcdf:
            self.retval_file.create_group ( "real_map" )
            self.retval_file.create_group ( "transformed_map" )

            oot = self.default_values.copy()
            oot.update ( self._unpack_to_dict ( r.x, do_invtransform=True ) )
            for k, v in oot.iteritems():
                self.retval_file.create_variable ( "real_map", k, v,
                                self.metadata.metadata[k].units, 
                                self.metadata.metadata[k].long_name,
                                self.metadata.metadata[k].std_name )
            oot = self.default_values.copy()
            oot.update ( self._unpack_to_dict ( r.x, do_invtransform=False ) )
            for k, v in oot.iteritems():
                self.retval_file.create_variable ( "transformed_map", k, v,
                                self.metadata.metadata[k].units, 
                                self.metadata.metadata[k].long_name,
                                self.metadata.metadata[k].std_name )
            if do_unc:
                self.retval_file.create_group ( "real_ci5pc" )
                self.retval_file.create_group ( "real_ci25pc" )
                self.retval_file.create_group ( "real_ci75pc" )
                self.retval_file.create_group ( "real_ci95pc" )
                self.retval_file.create_group ( "post_sigma" )
                unc_dict =  self.do_uncertainty ( r.x )
                for k, v in unc_dict.iteritems():
                    if k.find ("post_cov") >= 0 or k.find("hessian") >= 0 \
                        or k.find ( "post_sigma" ) >= 0:
                        print "Not done with this output yet (%s)" % k
                    else:
                        for kk, vv in v.iteritems():
                            self.retval_file.create_variable ( k, kk, vv,
                                self.metadata.metadata[kk].units, 
                                self.metadata.metadata[kk].long_name,
                                self.metadata.metadata[kk].std_name )

        if ret_sol or (not self.netcdf):
            
            retval_dict = {}
            retval_dict['real_map'] = self._unpack_to_dict ( r.x, do_invtransform=True )
            ### horrible HACK
            
            for k,v in self.state_config.iteritems():
                if self.invtransformation_dict.has_key ( k ) and v == FIXED:
                    retval_dict['real_map'][k] = self.default_values[k]
                    
            retval_dict['transformed_map'] = self._unpack_to_dict ( r.x, \
                do_invtransform=False )

            if do_unc:
                retval_dict.update ( self.do_uncertainty ( r.x ) )
            if self.verbose:
                print "Saving results to %s" % self.output_name
            cPickle.dump ( retval_dict, open( self.output_name, 'wb' ) )
        if ret_sol:
            return retval_dict
        else:
            return 0
    
    def do_uncertainty ( self, x ):
        """A method to calculate the uncertainty. Takes in a state vector.
        
        Parameters
        -----------
        x: array
            State vector (see ``_self._pack_to_dict``)
        
        Returns
        ---------
        A dictionary with the values for the posterior covariance function
        (sparse matrix), 5, 25, 75 and 95 credible intervals, and the
        main diagonal standar deviation. In each of these (apart from the
        posterior covariance sparse matrix), we get a new dictionary with
        parameter keys and the parameter estimation represented in the
        selected state grid."""
        
        
        the_hessian = sp.lil_matrix ( ( x.size, x.size ) )
        x_dict = self._unpack_to_dict ( x )
        #cost, der_cost = self.operators["Obs"].der_cost ( x_dict, \
            #self.state_config )
        #this_hessian = self.operators["Obs"].der_der_cost ( x_dict, \
                        #self.state_config, self, epsilon=1e-10 )
        
        #for epsilon in [ 10e-10, 1e-8, 1e-6, 1e-10, 1e-12, ]:
            # print "Hessian with epsilon=%e" % epsilon
        # epsilon is defined in order to use der_der_cost methods that
        # evaluate the Hessian numerically
        epsilon = 1e-8
        for op_name, the_op in self.operators.iteritems():
            # The try statement is here to allow der_der_cost methods to
            # take either a state dictionary or a state vector
            try:
               this_hessian = the_op.der_der_cost ( x_dict, \
                    self.state_config, self, epsilon=epsilon )
	    except OperatorDerDerTypeError:
                this_hessian = the_op.der_der_cost ( x, self.state_config, \
                    self, epsilon=epsilon )
            if self.verbose:
                print "Saving Hessian to %s_%s.pkl" % ( self.output_name, \
                    op_name )
            # Save the individual Hessian contributions to disk
            cPickle.dump ( this_hessian, open( "%s_%s_hessian.pkl" \
                % ( self.output_name, op_name ), 'w'))
            # Add the current Hessian contribution to the global Hessian
            the_hessian = the_hessian + this_hessian
        # Need to change the sparse storage format for the Hessian to do
        # the LU decomposition
        a_sps = sp.csc_matrix( the_hessian )
        # LU decomposition object
        lu_obj = sp.linalg.splu( a_sps )
        # Now, invert the Hessian in order to get the main diagonal elements
        # of the inverse Hessian (e.g. the variance)
        main_diag = np.zeros_like ( x )
        for k in xrange(x.size):
            b = np.zeros_like ( x )
            b[k] = 1
            main_diag[k] = lu_obj.solve ( b )[k]
            
        post_cov = sp.dia_matrix(main_diag,0 ).tolil() # Sparse purely diagonal covariance matrix 
        post_sigma = np.sqrt ( main_diag ).squeeze()
        # Calculate credible intervals, transform them back to real units, and 
        # store in a dictionary.
        _ci_5 = self._unpack_to_dict( x - 1.96*post_sigma, do_invtransform=True )
        _ci_95 = self._unpack_to_dict( x + 1.96*post_sigma, do_invtransform=True )
        _ci_25 = self._unpack_to_dict( x - 0.67*post_sigma, do_invtransform=True )
        _ci_75 = self._unpack_to_dict( x + 0.67*post_sigma, do_invtransform=True )
        # There intervals are OK in transformed space. However, we need to ensure that
        # e.g. ci_5 <= ci_95 in real coordinates. We do this in the next loop
        ci_5 = {}
        ci_95 = {}
        ci_25 = {}
        ci_75 = {}

        for k in self.state_config.iterkeys():
            ci_5[k]  = np.min(np.array([_ci_5[k],_ci_95[k]]),axis=0)
            ci_95[k] = np.max(np.array([_ci_5[k],_ci_95[k]]),axis=0)
            ci_25[k] = np.min(np.array([_ci_25[k],_ci_75[k]]),axis=0)
            ci_75[k] = np.max(np.array([_ci_25[k],_ci_75[k]]),axis=0)


        # Now store uncertainty, and return it to the user in a dictionary
        retval = {}
        retval['post_cov'] = post_cov
        retval['real_ci5pc'] = ci_5
        retval['real_ci95pc'] = ci_95
        retval['real_ci25pc'] = ci_25
        retval['real_ci75pc'] = ci_75
        retval['post_sigma'] = post_sigma
        retval['hessian'] =  the_hessian
        return retval
        
    def cost ( self, x ):
         """Calculate the cost function using a flattened state vector representation"""
         x_dict = self._unpack_to_dict ( x )
         # Store the parameter dictionary in case we need it later for e.g.
         # crossvalidation
         self.parameter_dictionary = x_dict
         aggr_cost = 0
         aggr_der_cost = x*0.0
         self.cost_components = {}
         start_time = time.time()
         for op_name, the_op in self.operators.iteritems():
             
             cost, der_cost = the_op.der_cost ( x_dict, self.state_config )
             aggr_cost = aggr_cost + cost
             aggr_der_cost = aggr_der_cost + der_cost
             self.cost_components[op_name] = der_cost
             if self.verbose:
                 print "\t%s %8.3e" % ( op_name, cost )
         self.the_cost = aggr_cost


         
         if self.verbose:
             print "Total cost: %8.3e" % aggr_cost
             print 'Elapsed: %.2f seconds' % (time.time() - start_time)
             
             
         return aggr_cost, aggr_der_cost
