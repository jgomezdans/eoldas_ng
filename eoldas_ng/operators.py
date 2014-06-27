#!/usr/bin/env python
"""
EOLDAS ng
==========

A reorganisation of the EOLDAS codebase

"""

__author__  = "J Gomez-Dans"
__version__ = "1.0 (1.12.2013)"
__email__   = "j.gomez-dans@ucl.ac.uk"

from collections import OrderedDict

import numpy as np
import scipy.optimize
import scipy.sparse as sp
from scipy.ndimage.interpolation import zoom

from eoldas_utils import *

FIXED = 1
CONSTANT = 2
VARIABLE = 3


class AttributeDict(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    



         
################################################################################        
################################################################################        

class Prior ( object ):
    """A gaussian prior class"""
    def __init__ ( self, prior_mu, prior_inv_cov ):
        """The prior constructor.
        
        We take a dictionary with means and inverse covariance structures. The elements
        of the dictionary can either be 1-element or several element arrays (so you 
        could have an estimate of LAI for each time step from climatology, or a
        single value). The inverse covariance (or precision) matrix is either a single
        value ($1/\sigma^{2}$), or a full matrix. If you pass a single value for a
        VARIABLE parameter, it will be converted into a diagonal matrix automatically
        
        NOTE that if **transformed variables** are used, the prior needs to be in transformed
        units, not in real units. 
        """
        self.mu = prior_mu
        self.inv_cov = prior_inv_cov
    def pack_from_dict ( self, x_dict, state_config ):
        """This method returns a vector from a dictionary and state configuration
        object. The idea is to use this with the prior sparse represntation, to
        get speed gains."""
        n, n_elems = get_problem_size ( x_dict, state_config )
        the_vector = np.zeros ( n )
        # Now, populate said vector in the right order
        # looping over state_config *should* preserve the order
        i = 0
        for param, typo in state_config.iteritems():
            if typo == CONSTANT: # Constant value for all times
                the_vector[i] = x_dict[param]
                i = i+1        
            elif typo == VARIABLE:
                # For this particular date, the relevant parameter is at location iloc
                the_vector[i:(i + n_elems)] =  \
                        x_dict[param].flatten() 
                i += n_elems
        return the_vector 
        
                    
    def first_guess ( self, state_config, n_elems ):
        """This method provides a simple way to initialise the optimisation: when called
        with a `state_config` dictionary, it will produce a starting point dictionary that
        can be used for the optimisation. We also need `n_elems`, the number of elements
        of the state for the VARIABLE parameters
        
        Parameters
        ----------
        state_config:dict
            A state configuration ordered dictionary
        n_elems: int
            The number of elements for VARIABLE state components
        
        Returns
        -------
        x0: dict
            A dictionary that can the be used as a starting point for optimisation.
        """
        
        x0 = dict()
        for param, typo in state_config.iteritems():
            
            if typo == FIXED: # Default value for all times
                # Doesn't do anything so we just skip
                pass        
            elif typo == CONSTANT: # Constant value for all times
                x0[param] = self.mu[param]
            elif typo == VARIABLE:
                x0[param] = np.ones( n_elems )*self.mu[param]
                
        return x0        
        
    def der_cost ( self, x_dict, state_config ):
        """Calculate the cost function and its partial derivatives for the prior object.
        Assumptions of normality are clear# Constant value for all times
        Takes a parameter dictionary, and a state configuration dictionary as
        inputs.
        
        Parameters
        -----------
        x_dict: ordered dict
            The state as a dictionary
        state_config: oredered dict
            The configuration dictionary
        
        Returns
        --------
        cost: float
            The value of the cost function
        der_cost: array
            An array with the partial derivatives of the cost function
        """
        if sp.issparse ( self.inv_cov ):
            x = self.pack_from_dict ( x_dict, state_config )
            err = sp.lil_matrix ( x - self.mu )
            cost = err.dot ( self.inv_cov ).dot ( err.T )
            der_cost  = np.array( err.dot ( self.inv_cov ).todense()).squeeze()
            cost = float(np.array(cost.todense()).squeeze())
            return cost, der_cost

        # Find out about problems size
        n, n_elems = get_problem_size ( x_dict, state_config )
        # Allocate space/initialise the outputs
        der_cost = np.zeros ( n )
        cost = 0
        # The next loop calculates the cost and associated partial derivatives
        # Mainly based on the parameter type
        i = 0 # Loop variable
        for param, typo in state_config.iteritems():
            if typo == FIXED: 
                # Doesn't do anything so we just skip
                pass
            elif typo == CONSTANT: 
                cost = cost + 0.5*( x_dict[param] - \
                            self.mu[param])**2*self.inv_cov[param]
                der_cost[i] = ( x_dict[param] - self.mu[param]) * \
                            self.inv_cov[param]
                i += 1                
            elif typo == VARIABLE:
                
                if self.inv_cov[param].size == 1:
                    # Single parameter for all sites/locations etc
                    # This should really be in the __init__ method!
                    sigma = self.inv_cov[param]
                    
                    self.inv_cov[param] = sp.dia_matrix ( ( np.ones(n_elems)*sigma, 0 ), shape=(n_elems, n_elems))

                
                cost_m = ( x_dict[param].flatten() - self.mu[param]) * ( \
                            self.inv_cov[param] )
                cost = cost + 0.5*(cost_m*(x_dict[param].flatten() - \
                            self.mu[param])).sum()
                der_cost[i:(i+n_elems)] = cost_m                                         
                
                i += n_elems
        
        return cost, der_cost

    def der_der_cost ( self, x_dict, state_config, state, epsilon=None ):
        """ The Hessian is just the inverse prior covariance matrix.
        However, we require the extra parameters for consistency, and
        to work out the positioning of the Hessian elements. The returned
        matrix is LIL-sparse.
        
        In the case the user has provided a sparse prior inverse matrix,
        we can just return this.
                
        Parameters
        -----------
        x_dict: ordered dict
            The state as a dictionary
        state_config: oredered dict
            The configuration dictionary
        
        Returns
        -------
        Hess: sparse matrix
            The hessian for the cost function at `x`
        """
        if sp.issparse ( self.inv_cov ):
            # We already have it!!!
            return self.inv_cov
        n_blocks = 0 # Blocks in sparse Hessian matrix
        for param, typo in state_config.iteritems():
            if typo == CONSTANT:
                n_blocks += 1
            elif typo == VARIABLE:
                n_blocks += 1

        n, n_elems = get_problem_size ( x_dict, state_config )
        block_mtx = []   
        i = 0
        jj = 0
        for param, typo in state_config.iteritems():
            
            if typo == FIXED:
                pass
            elif typo == CONSTANT:
                this_block = [ None for i in xrange( n_blocks) ]
                this_block[jj] = sp.lil_matrix ( self.inv_cov[param] )
                block_mtx.append ( this_block )
                jj += 1
                i += 1
            elif typo == VARIABLE:
                this_block = [ None for i in xrange( n_blocks) ]
                this_block[jj] = self.inv_cov[param]
                block_mtx.append ( this_block )

                jj += 1
                #h1[i:(i+n_elems), i:(i+n_elems)] = self.inv_cov[param].tolil() 
                i += n_elems
        # Typically, the matrix wil be sparse. In fact, in many situations,
        # it'll be purely diagonal, but in general, LIL is a good format
        return sp.bmat ( block_mtx, format="lil", dtype=np.float32 )
        
    
    


class TemporalSmoother ( object ):
    """A temporal smoother class"""
    def __init__ ( self, state_grid, gamma, order=1, required_params = None  ):
        self.order = order
        self.n_elems = state_grid.shape[0]
        I = np.identity( state_grid.shape[0] )
        self.D1 = np.matrix(I - np.roll(I,1))
        self.D1 = self.D1 * self.D1.T
        
        self.required_params = required_params
        if required_params is not None:
            n_reg_params = len ( required_params )
            if np.size ( gamma ) == 1:
                self.gamma = gamma*np.ones ( n_reg_params )
            else:
                self.gamma = gamma
        
    def der_cost ( self, x_dict, state_config):
        """
        Calculate the cost function and its partial derivs for a temporal smoother.
        Takes a parameter state dictionary, and a state configuration dictionary as
        inputs.
        
                
        Parameters
        -----------
        x_dict: ordered dict
            The state as a dictionary
        state_config: oredered dict
            The configuration dictionary
        
        Returns
        ---------
        cost: float
            The value of the cost function
        der_cost: array
            An array with the partial derivatives of the cost function
        """
        i = 0
        cost = 0
        n = 0
        self.required_params = self.required_params or state_config.keys()
        
        n, n_elems = get_problem_size ( x_dict, state_config )        
        der_cost = np.zeros ( n )
        isel_param = 0
        for param, typo in state_config.iteritems():
            
            if typo == FIXED: 
                pass
            if typo == CONSTANT: 
                # No model constraint!
                i += 1                
            elif typo == VARIABLE:
                if param in self.required_params :
                    xa = np.matrix ( x_dict[param] )
                    cost = cost + \
                        0.5*self.gamma[isel_param]*(np.sum(np.array(self.D1.dot(xa.T))**2))
                    der_cost[i:(i+self.n_elems)] = np.array( \
                        self.gamma[isel_param]*np.dot((self.D1).T, \
                        self.D1*np.matrix(xa).T)).squeeze()
                    isel_param += 1
                i += self.n_elems
                
                
        return cost, der_cost
    
    def der_der_cost ( self, x, state_config, state, epsilon=None ):
        """ The Hessian for this cost function is determined analytically, but
        we need some additional parameters for consistency, and
        to work out the positioning of the Hessian elements. The returned
        matrix is LIL-sparse (Mostly, it's multidiagonal).
        
                
        Parameters
        -----------
        x_dict: ordered dict
            The state as a dictionary
        state_config: oredered dict
            The configuration dictionary
        state: State
            The state is required in some cases to gain access to parameter
            transformations.
        Returns
        ---------
        Hess: sparse matrix
            The hessian for the cost function at `x`
        """
        n = 0
        for param, typo in state_config.iteritems():
            if typo == CONSTANT:
                n += 1
            elif typo == VARIABLE:
                n_elems = x[param].size
                n += n_elems
        
        h = sp.lil ( (n ,n ) )
        i = 0
        isel_param = 0
        for param, typo in state_config.iteritems():
            
            if typo == FIXED: # Default value for all times
                # Doesn't do anything so we just skip
                pass
                
            elif typo == CONSTANT: # Constant value for all times
                # h[i, i ] = 0.0 Matrix is sparse now ;-)
                i += 1                
                
            elif typo == VARIABLE:
                if param in self.required_params:
                    hessian = sp.lil_matrix ( self.gamma[isel_param]*np.dot ( \
                         self.D1,np.eye( self.n_elems )).dot( self.D1.T ) )
                    h[i:(i+n_elems), i:(i+n_elems) ] = hessian
                    isel_param += 1
                    i += n_elems
        return h

class SpatialSmoother ( object ):
    """MRF prior"""
    def __init__ ( self, state_grid, gamma, required_params = None  ):
        self.nx = state_grid.shape
        self.gamma = gamma
        self.required_params = required_params
        
    def der_cost ( self, x_dict, state_config):
        """Calculate the cost function and its partial derivs for a spatial 
        constraint (aka Markov Random Field, MRF)
        
                
        Parameters
        -----------
        x_dict: ordered dict
            The state as a dictionary
        state_config: oredered dict
            The configuration dictionary
        
        Returns
        --------
        cost: float
            The value of the cost function
        der_cost: array
            An array with the partial derivatives of the cost function
        """

        i = 0
        cost = 0
        n = 0
        self.required_params = self.required_params or state_config.keys()
        
        n, n_elems = get_problem_size ( x_dict, state_config )
        
        der_cost = np.zeros ( n )

        for param, typo in state_config.iteritems():
            
            if typo == FIXED: # Default value for all times
                # Doesn't do anything so we just skip
                pass
                
            if typo == CONSTANT: # Constant value for all times
                # No model constraint!
                               
                i += 1                
                
            elif typo == VARIABLE:
                if param in self.required_params :
                    
                    try:
                        sigma_model = self.gamma[ \
                            self.required_params.index(param) ]
                    except:
                        sigma_model = self.gamma
                   
                    xa = x_dict[param].reshape( self.nx )
                    cost, dcost = fit_smoothness ( xa, sigma_model )
                    der_cost[i:(i+n_elems)] = dcost.flatten()
                i += n_elems
                
                
        return cost, der_cost
    
    def der_der_cost ( self, x, state_config, state, epsilon=None ):
        # TODO Clear how it goes for single parameter, but for
        # multiparameter, it can easily get tricky. Also really
        # need to have all mxs in sparse format, otherwise, they
        # don't fit in memory.
        block_mtx = []
        n = 0
        n_blocks = 0 # Blocks in sparse Hessian matrix
        for param, typo in state_config.iteritems():
            if typo == CONSTANT:
                n += 1
                n_blocks += 1
            elif typo == VARIABLE:
                n_elems = x[param].size
                n += n_elems
                n_blocks += 1
        #h = sp.lil_matrix ( (n ,n ), dtype=np.float32 )
        rows, cols = self.nx # Needs checking...
        jj = 0
        for param, typo in state_config.iteritems():    
            if typo == FIXED: # Default value for all times
                # Doesn't do anything so we just skip
                pass   
                
            if typo == CONSTANT: # Constant value for all times
                # No model constraint!            
                this_block = [ None for i in xrange(n_blocks) ]
                this_block [jj] = sp.lil_matrix(np.array([[0.]]))
                block_mtx.append ( this_block )
                jj += 1          

            elif typo == VARIABLE:
                if param in self.required_params :
                    try:
                        sigma_model = self.gamma[param]
                    except:
                        sigma_model = self.gamma

                    # Generate DeltaY
                    # The first diagonal is defined as...
                    d1 = np.ones(rows*rows, dtype=np.int8)*2 
                    d1[::rows] = 1 
                    d1 [(rows-1)::rows] = 1
                    # The +1 or -1 diagonals are
                    d2 = -np.ones(rows*rows, dtype=np.int8) 
                    d2[(rows-1)::rows] = 0
                    DYsyn = sp.dia_matrix ( (d1,0), shape=(rows*cols, rows*cols)) + \
                             sp.dia_matrix ( (np.r_[0,d2], 1), shape=(rows*cols, rows*cols)) + \
                             sp.dia_matrix ( (np.r_[d2, 0], -1), shape=(rows*cols, rows*cols))
                    DYsparse = scipy.sparse.dia_matrix (DYsyn, dtype=np.float32)

                    #Generate DeltaX
                    # The main diagonal
                    d1 = 2*np.ones(rows*rows, dtype=np.int8) 
                    d1[:rows] = 1 
                    d1[-rows:] = 1


                    d2 = -1*np.ones(rows*rows, dtype=np.int8)
                    DXsparse = scipy.sparse.spdiags( [ d1, d2, d2], \
                        [0, rows, -rows], rows*rows, rows*rows)
                    # Stuff this particular bit of the Hessian in the complete
                    # big matrix...
                    this_block = [ None for i in xrange(n_blocks) ]
                    this_block [jj] = ((DYsparse + DXsparse)/\
                                    sigma_model**2)
                    block_mtx.append ( this_block )
                    jj += 1          

                    
        ### h neds to be defined as a sp.bmat, and build from the individual
        ### sparse blocks, including for single parameters (these are 0 contributions)
        ### need to produce a list like this:
        ### [ [ h None None],[ None, H, None]] etc.
        h = sp.bmat ( block_mtx, format="lil", dtype=np.float32 )
        return h
        
class ObservationOperator ( object ):
    """An Identity observation operator"""
    def __init__ ( self, state_grid, observations, sigma_obs, mask, \
		required_params = ['magnitude'], factor=1 ):
        self.observations = observations
        self.sigma_obs = sigma_obs
        self.mask = mask
        self.n_elems = state_grid.size
        self.required_params = required_params
        self.factor = factor
        
    def der_cost ( self, x_dict, state_config ):
        """Calculate the cost function and its partial derivs for a identity
        observation operator (i.e., where the observations are the same magnitude
        as the state)
                
        Parameters
        -----------
        x_dict: ordered dict
            The state as a dictionary
        state_config: oredered dict
            The configuration dictionary
        
        Returns
        --------
        cost: float
            The value of the cost function
        der_cost: array
            An array with the partial derivatives of the cost function
        """
        i = 0
        cost = 0
        n = 0
        # n is used to calculate the size of the derivative vector. 
        # This should be part of the state
        # TODO pass the der_cost array and modify in-place!
        
        for typo in x_dict.iteritems():
            if np.isscalar ( typo[1] ):
                n = n + 1
            else:
                n = n + len ( typo[1] )
        der_cost = np.zeros ( self.n_elems )
        for param, typo in state_config.iteritems():
            
            if typo == FIXED: # Default value for all times
                # Doesn't do anything so we just skip
                pass
                
            if typo == CONSTANT: # Constant value for all times
                # No model constraint!
                               
                i += 1                
                
            elif typo == VARIABLE:
                if param in self.required_params:
                    this_cost, this_der = fit_observations_gauss ( x_dict[param], \
                        self.observations, self.sigma_obs, self.mask, factor=self.factor )
                    cost = cost + this_cost
                    der_cost[i:(i+self.n_elems)] = this_der.flatten()
                i += self.n_elems
                
        return cost, der_cost

    def der_der_cost ( self, x_dict, state_config, state, epsilon=None ):
        # Hessian is just C_{obs}^{-1}?
        n, n_elems = get_problem_size ( x_dict, state_config )
        h1 = np.zeros ( n )
        h1[ self.mask ] = (1./self.sigma_obs**2)
        return sp.lil_matrix (  np.diag( h1 ) )
        

        
class ObservationOperatorTimeSeriesGP ( object ):
    """A GP-based observation operator"""
    def __init__ ( self, state_grid, state, observations, mask, emulators, bu, \
            band_pass=None, bw=None ):
        """
         observations is an array with n_bands, nt observations. nt has to be the 
         same size as state_grid (can have dummny numbers in). mask is nt*4 
         (mask, vza, sza, raa) array.
         
         
        """
        self.state = state
        self.observations = observations
        try:
            self.n_obs, self.n_bands = self.observations.shape
        except:
            raise ValueError, "Typically, obs should be (n_obs, n_bands)"
        self.mask = mask
        assert ( self.n_obs ) == mask.shape[0]
        self.state_grid = state_grid
        self.nt = self.state_grid.shape[0]
        self.emulators = emulators
        self.bu = bu
        self.band_pass = band_pass
        self.bw = bw

        
    
    def der_cost ( self, x_dict, state_config ):

        """The cost function and its partial derivatives. One important thing
        to note is that GPs have been parameterised in transformed space, 
        whereas `x_dict` is in "real space". So when we go over the parameter
        dictionary, we need to transform back to linear units. TODO Clearly, it
        might be better to have cost functions that report whether they need
        a dictionary in true or transformed units!

        Parameters
        -----------
        x_dict: ordered dict
            The state as a dictionary
        state_config: oredered dict
            The configuration dictionary
        
        Returns
        --------
        cost: float
            The value of the cost function
        der_cost: array
            An array with the partial derivatives of the cost function        
        """
        i = 0
        cost = 0.
        n, n_elems = get_problem_size ( x_dict, state_config )
        der_cost = np.zeros ( n )
        x_params = np.empty ( ( len( x_dict.keys()), \
                self.nt ) )
        j = 0
        ii = 0
        
        the_derivatives = np.zeros ( ( len( x_dict.keys()), \
                self.nt ) )
        for param, typo in state_config.iteritems():
        
            if typo == FIXED or  typo == CONSTANT:
                #if self.state.transformation_dict.has_key ( param ):
                    #x_params[ j, : ] = self.state.transformation_dict[param] ( x_dict[param] )
                #else:
                x_params[ j, : ] = x_dict[param]
                
            elif typo == VARIABLE:
                #if self.state.transformation_dict.has_key ( param ):
                    #x_params[ j, : ] = self.state.transformation_dict[param] ( x_dict[param] )
                #else:
                x_params[ j, : ] = x_dict[param]
            j += 1
        self.fwd_modelled_obs = []
        istart_doy = self.state_grid[0]
        for itime, tstep in enumerate ( self.state_grid[1:] ):
            # Select all observations between istart_doy and tstep
            sel_obs = np.where ( np.logical_and ( self.mask[:, 0] > istart_doy, \
                self.mask[:, 0] <= tstep ), True, False )
            if sel_obs.sum() == 0:
                # We have no observations, go to next period!
                istart_doy = tstep # Update istart_doy
                continue
            # Now, test the QA flag, field 2 of the mask...
            sel_obs = np.where ( np.logical_and ( self.mask[:, 1], sel_obs ), \
                True, False )
            if sel_obs.sum() == 0:
                # We have no observations, go to next period!
                istart_doy = tstep # Update istart_doy
                continue
            # In this bit, we need a loop to go over this period's observations
            # And add the cost/der_cost contribution from each.
            for this_obs_loc in sel_obs.nonzero()[0]:
                this_obsop, this_obs, this_extra = self.time_step ( \
                    this_obs_loc )
                this_cost, this_der, fwd_model = self.calc_mismatch ( this_obsop, \
                    x_params[:, itime], \
                    this_obs, self.bu, *this_extra )
                self.fwd_modelled_obs.append ( fwd_model ) # Store fwd model
                cost += this_cost
                the_derivatives[ :, itime] += this_der
            # Advance istart_doy to the end of this period
            istart_doy = tstep
            
        j = 0
        for  i, (param, typo) in enumerate ( state_config.iteritems()) :
            if typo == CONSTANT:
                der_cost[j] = the_derivatives[i, self.mask[:,1] != 0].sum()
                j += 1
            elif typo == VARIABLE:
                n_elems = x_dict[param].size
                der_cost[j:(j+n_elems) ] = the_derivatives[i, :]
                j += n_elems
        
        return cost, der_cost
    
    def time_step ( self, this_loc ):
        """Returns relevant information on the observations for a particular time step.
        """
        tag = np.round( self.mask[ this_loc, 2:].astype (np.int)/5.)*5
        tag = tuple ( (tag[:2].astype(np.int)).tolist() )
        this_obs = self.observations[ this_loc, :]
        return self.emulators[tag], this_obs, [ self.band_pass, self.bw ]
    
    def calc_mismatch ( self, gp, x, obs, bu, band_pass, bw ):
        this_cost, this_der, fwd = fwd_model ( gp, x, obs, bu, band_pass, bw )
        return this_cost, this_der, fwd
    
    
    def der_der_cost ( self, x_dict, state_config, state, epsilon=1.0e-5 ):
        """Numerical approximation to the Hessian. This approximation is quite
        simple, and is based on a finite differences of the individual terms of 
        the cost function. Note that this method shares a lot with the `der_cost`
        method in the same function, and a refactoring is probably required, or
        even better, a more "analytic" expression making use of the properties of
        GPs to calculate the second derivatives.
                
                
        Parameters
        -----------
        x_dict: ordered dict
            The state as a dictionary
        state_config: oredered dict
            The configuration dictionary
        state: State
            The state is required in some cases to gain access to parameter
            transformations.
        Returns
        ---------
        Hess: sparse matrix
            The hessian for the cost function at `x`
        """
        
        
        i = 0
        cost = 0.
        
        n, n_elems = get_problem_size ( x_dict, state_config )
        
        der_cost = np.zeros ( n )
        h = sp.lil_matrix ( (n,n))
        x_params = np.empty ( ( len( x_dict.keys()), self.nt ) )
        j = 0
        ii = 0
        the_derivatives = np.zeros ( ( len( x_dict.keys()), self.nt ) )
        param_pattern = np.zeros ( len( state_config.items()))
        for param, typo in state_config.iteritems():
        
            if typo == FIXED:  
                #if self.state.transformation_dict.has_key ( param ):
                    #x_params[ j, : ] = self.state.transformation_dict[param] ( x_dict[param] )
                #else:
                x_params[ j, : ] = x_dict[param]
                param_pattern[j] = FIXED
            elif typo == CONSTANT:
                x_params[ j, : ] = x_dict[param]
                param_pattern[j] = CONSTANT
                
            elif typo == VARIABLE:
                #if self.state.transformation_dict.has_key ( param ):
                    #x_params[ j, : ] = self.state.transformation_dict[param] ( x_dict[param] )
                #else:
                x_params[ j, : ] = x_dict[param]
                param_pattern[j] = VARIABLE

            j += 1
        
        n_const = np.sum ( param_pattern == CONSTANT )
        n_var = np.sum ( param_pattern == VARIABLE )
        n_grid = self.nt # don't ask...
        istart_doy = self.state_grid[0]
        for itime, tstep in enumerate ( self.state_grid[1:] ):
            # Select all observations between istart_doy and tstep
            sel_obs = np.where ( np.logical_and ( self.mask[:, 0] > istart_doy, \
                self.mask[:, 0] <= tstep ), True, False )
            if sel_obs.sum() == 0:
                # We have no observations, go to next period!
                istart_doy = tstep # Update istart_doy
                continue
            # Now, test the QA flag, field 2 of the mask...
            sel_obs = np.where ( np.logical_and ( self.mask[:, 1], sel_obs ), \
                True, False )
            if sel_obs.sum() == 0:
                # We have no observations, go to next period!
                istart_doy = tstep # Update istart_doy
                continue
            # In this bit, we need a loop to go over this period's observations
            # And add the cost/der_cost contribution from each.
            for this_obs_loc in sel_obs.nonzero()[0]:
                
                this_obsop, this_obs, this_extra = self.time_step ( \
                    this_obs_loc )
                xs = x_params[:, itime]*1
                dummy, df_0 = self.calc_mismatch ( this_obsop, \
                    xs, this_obs, self.bu, *this_extra )
                iloc = 0
                iiloc = 0
                for i,fin_diff in enumerate(param_pattern):
                    if fin_diff == 1: # FIXED
                        continue                    
                    xxs = xs[i]*1
                    xs[i] += epsilon
                    dummy, df_1 = self.calc_mismatch ( this_obsop, \
                        xs, this_obs, self.bu, *this_extra )                    # Calculate d2f/d2x
                    hs =  (df_1 - df_0)/epsilon
                    if fin_diff == 2: # CONSTANT
                        iloc += 1
                    elif fin_diff == 3: # VARIABLE
                        iloc = n_const + iiloc*n_grid + itime
                        iiloc += 1
                    jloc = 0
                    jjloc = 0
                    for j,jfin_diff in enumerate(param_pattern):
                        if jfin_diff == FIXED: 
                            continue
                        if jfin_diff == CONSTANT: 
                            jloc += 1
                        elif jfin_diff == VARIABLE: 
                            jloc = n_const + jjloc*n_grid + itime
                            jjloc += 1
                        h[iloc, jloc] += hs[j]     
                    xs[i] = xxs
            # Advance istart_doy to the end of this period
            istart_doy = tstep

        return sp.lil_matrix ( h.T )
        




         
class ObservationOperatorImageGP ( object ):
    """A GP-based observation operator"""
    def __init__ ( self, state_grid, state, observations, mask, emulators, bu, \
            factor=None, band_pass=None, bw=None, per_band=False ):
        """
         observations is an array with n_bands, nt observations. nt has to be the 
         same size as state_grid (can have dummny numbers in). mask is nt*4 
         (mask, vza, sza, raa) array.
             
         
        """
        self.state = state
        self.observations = observations
        try:
            self.n_bands, self.nx, self.ny = self.observations.shape
        except:
            raise ValueError, "Typically, obs should be n_bands * nx * ny"
        self.mask = mask
        assert observations.shape[1:] == mask.shape
        self.state_grid = state_grid
        self.nx_state, self.ny_state = state_grid.shape
        self.original_emulators = emulators # Keep around for quick inverse emulators
        if per_band:
            if band_pass is None:
                raise IOError, \
                    "You want fast emulators, need to provide bandpass fncs!"
            self.emulators = perband_emulators ( emulators, band_pass )
            self.per_band = True
        
        else:
            self.per_band = False
            self.emulators = emulators
        self.bu = bu
        self.band_pass = band_pass
        self.bw = bw
        self.factor = factor
        self.fwd_modelled_obs = np.zeros_like ( self.observations )

    def first_guess ( self, state_config ):
        """
        A method to provide a first guess of the state. The idea here is to take the GPs, 
        and recast them, so that rather than provide an emulator, they provide a regressor
        from input reflectance/radiance etc to surface parameters
        """
        
        gps = create_inverse_emulators ( self.original_emulators, \
            self.band_pass, state_config )

        x0 = dict()        
        for param, gp in gps.iteritems():
            x0[param] = np.zeros_like( self.observations[0,:, :].flatten())
            x0[param][self.mask.flatten()] = gp.predict( self.observations[:, self.mask].T )[0]
            x0[param][~self.mask.flatten()] = x0[param][self.mask.flatten()].mean()
        return x0

    def predict_observations ( self, x_dict, state_config ):
        """
        This method forward models the observations based on the state dict. The
        idea is to predict the entire observations for the sensor, as a check
        that e.g. the assimilation worked, but also for cross-validation, as
        the masks will not be used (i.e., we predict all pixels, even those that
        have been masked)
        
                
        Parameters
        -----------
        x_dict: ordered dict
            The state as a dictionary
        state_config: oredered dict
            The configuration dictionary
        
        Returns
        --------
        predictions: array
            An array identical to self.observations, with a prediction based on
            the chosen `x_dict`.

        """
        i = 0
        cost = 0.
        n = 0
        n = 0
        for param, typo in state_config.iteritems():
            if typo == CONSTANT:
                n += 1
            elif typo == VARIABLE:
                n_elems = x_dict[param].size
                n += n_elems
        der_cost = np.zeros ( n )
        # `x_params` should relate to the grid state size, not observations size
        x_params = np.empty ( ( len( x_dict.keys()), \
            self.nx_state * self.ny_state ) )
        j = 0
        ii = 0
        # `the_derivatives` should relate to the grid state size, not 
        # observations size
        the_derivatives = np.zeros ( ( len( x_dict.keys()), \
                self.nx_state * self.ny_state ) )
        for param, typo in state_config.iteritems():
        
            if typo == FIXED or  typo == CONSTANT:
                #if self.state.transformation_dict.has_key ( param ):
                    #x_params[ j, : ] = self.state.transformation_dict[param] ( x_dict[param] )
                #else:
                x_params[ j, : ] = x_dict[param]
                
            elif typo == VARIABLE:
                #if self.state.transformation_dict.has_key ( param ):
                    #x_params[ j, : ] = self.state.transformation_dict[param] ( x_dict[param] )
                #else:
                x_params[ j, : ] = x_dict[param].flatten()

            j += 1
        
        # x_params is [n_params, Nx*Ny]
        predictions = self.observations * 0.
        for band in xrange ( self.n_bands ):
            # Run the emulator forward. Doing it for all pixels, or only for
            # the unmasked ones
            # Also, need to work out whether the size of the state is 
            # different to that of the observations (ie integrate over coarse res data)
            fwd_model, emu_err, partial_derv = \
                self.emulators[band].predict ( x_params[:, :].T )
            if self.factor is not None:
                # Multi-resolution! Need to integrate over the low resolution
                # footprint using downsample in `eoldas_utils`
                fwd_model = downsample ( fwd_model.reshape( \
                    self.state_grid.shape), self.factor[0], \
                    self.factor[1] ).flatten()
            # Now calculate the cost increase due to this band...
            predictions [ band, :, : ] = fwd_model.reshape( (self.mask.shape))
        return predictions




        
    def der_cost ( self, x_dict, state_config ):

        """
        The cost function and its partial derivatives. One important thing
        to note is that GPs have been parameterised in transformed space, 
        whereas `x_dict` is in "real space". So when we go over the parameter
        dictionary, we need to transform back to linear units. TODO Clearly, it
        might be better to have cost functions that report whether they need
        a dictionary in true or transformed units!
        
        Parameters
        -----------
        x_dict: ordered dict
            The state as a dictionary
        state_config: oredered dict
            The configuration dictionary
        
        Returns
        --------
        cost: float
            The value of the cost function
        der_cost: array
            An array with the partial derivatives of the cost function
        """
        i = 0
        cost = 0.
        n = 0
        n = 0
        for param, typo in state_config.iteritems():
            if typo == CONSTANT:
                n += 1
            elif typo == VARIABLE:
                n_elems = x_dict[param].size
                n += n_elems
        der_cost = np.zeros ( n )
        # `x_params` should relate to the grid state size, not observations size
        x_params = np.empty ( ( len( x_dict.keys()), \
            self.nx_state * self.ny_state ) )
        j = 0
        ii = 0
        # `the_derivatives` should relate to the grid state size, not 
        # observations size
        the_derivatives = np.zeros ( ( len( x_dict.keys()), \
                self.nx_state * self.ny_state ) )
        for param, typo in state_config.iteritems():
        
            if typo == FIXED or  typo == CONSTANT:
                #if self.state.transformation_dict.has_key ( param ):
                    #x_params[ j, : ] = self.state.transformation_dict[param] ( x_dict[param] )
                #else:
                x_params[ j, : ] = x_dict[param]
                
            elif typo == VARIABLE:
                #if self.state.transformation_dict.has_key ( param ):
                    #x_params[ j, : ] = self.state.transformation_dict[param] ( x_dict[param] )
                #else:
                x_params[ j, : ] = x_dict[param].flatten()

            j += 1
        
        # x_params is [n_params, Nx*Ny]
        # it should be able to run the emulators directly on x_params, and then 
        # do a reshape
        if self.factor is not None:
            # Interpolate the mask is needed for the derivatives
            zmask = zoom ( self.mask, self.factor, order=0, mode="nearest" \
                ).astype ( np.bool )
        else:
            zmask = self.mask

        for band in xrange ( self.n_bands ):
            # Run the emulator forward. Doing it for all pixels, or only for
            # the unmasked ones
            # Also, need to work out whether the size of the state is 
            # different to that of the observations (ie integrate over coarse res data)
            fwd_model, emu_err, partial_derv = \
                self.emulators[band].predict ( x_params[:, zmask.flatten()].T )
            if self.factor is not None:
                # Multi-resolution! Need to integrate over the low resolution
                # footprint using downsample in `eoldas_utils`
                fwd_model = downsample ( fwd_model.reshape( \
                    self.state_grid.shape), self.factor[0], \
                    self.factor[1] ).flatten()
            # Now calculate the cost increase due to this band...
            err = ( fwd_model - self.observations[band, self.mask] )
            self.fwd_modelled_obs[band, self.mask] = fwd_model
            cost += np.sum(0.5 * err**2/self.bu[band]**2 )
            # And update the partial derivatives
            #the_derivatives += (partial_derv[self.mask.flatten(), :] * \
                #(( fwd_model[self.mask.flatten()] - \
                #self.observations[band, self.mask] ) \
                #/self.bu[band]**2)[:, None]).T
            
            # TODO there's a mismatch of sizes. The derivatives are in state
            # grid, not in observation grid. So we need to "zoom" the second
            # line of the following expression (fwd_model - obs) to be the
            # same shape as the derivatives
            # TODO also note how we cope with data gaps here, as the mask also
            # appears on the RHS of the expression. Do I also need a zoomed
            # version of the mask?
            if self.factor is not None:
                err = zoom ( err.reshape((self.nx, self.ny)), \
                    self.factor, order=0, mode="nearest" ).flatten()
 
            the_derivatives[:, zmask.flatten()] += (partial_derv[:, :] * \
                (err/self.bu[band]**2)[:, None]).T
            ####the_derivatives[:, self.mask.flatten()] += (partial_derv[:, :] * \
                ####(( fwd_model[:] - \
                ####self.observations[band, self.mask] ) \
                ####/self.bu[band]**2)[:, None]).T

        
        j = 0
        for  i, (param, typo) in enumerate ( state_config.iteritems()) :
            if typo == CONSTANT:
                der_cost[j] = the_derivatives[i, 0]
                j += 1
            elif typo == VARIABLE:
                n_elems = x_dict[param].size
                der_cost[j:(j+n_elems) ] = the_derivatives[i, :]
                j += n_elems
        
        return cost, der_cost
    
    def der_der_cost ( self, x, state_config, state, epsilon=1.0e-5 ):
        """Numerical approximation to the Hessian"""

        N = x.size
        h = np.zeros((N,N))
        x_dict = state._unpack_to_dict ( x, do_invtransform=True )
        df_0 = self.der_cost ( x_dict, state_config )[1]
        for i in xrange(N):
            xx0 = 1.*x[i]
            x[i] = xx0 + epsilon
            x_dict = state._unpack_to_dict ( x, do_invtransform=True )
            df_1 = self.der_cost ( x_dict, state_config )[1]
            h[i,:] = (df_1 - df_0)/epsilon
            x[i] = xx0
        return sp.lil_matrix ( h )
