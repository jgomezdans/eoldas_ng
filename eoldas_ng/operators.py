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
        
                    
    def first_guess ( self, state_config, n_elems, do_unc=False ):
        """This method provides a simple way to initialise the optimisation: when called
        with a `state_config` dictionary, it will produce a starting point dictionary that
        can be used for the optimisation. We also need `n_elems`, the number of elements
        of the state for the VARIABLE parameters
        
        Parameters
        ----------
        state_config: dict
            A state configuration ordered dictionary
        n_elems: int
            The number of elements for VARIABLE state components
        do_unc: bool
            Whether to return the uncertainty
        Returns
        -------
        x0: dict
            A dictionary that can the be used as a starting point for optimisation.
        """
        
        # **If** we have a sparse inverse covariance, we need to fish out the
        # elements of both the prior and the covariance. If not, we just report
        # things as before
        
        if sp.issparse ( self.inv_cov ):
            #raise NotImplementedError( "The prior inverse covariance + " + \
                #"matrix is sparse!" )
            return self.mu
            
            
        else:    
            x0 = dict()
            for param, typo in state_config.iteritems():
                
                if typo == FIXED: # Default value for all times
                    # Doesn't do anything so we just skip
                    pass        
                elif typo == CONSTANT: # Constant value for all times
                    if do_unc:
                        x0[param] = self.mu[param]
                        s0[param] = 1./self.inv_cov[param]
                    else:
                        x0[param] = self.mu[param]
                elif typo == VARIABLE:
                    if do_unc:
                        x0[param] = np.ones( n_elems ) * self.mu[param]
                        s0[param] = np.ones( n_elems ) * 1./self.inv_cov[param]
                    else:
                        x0[param] = np.ones( n_elems ) * self.mu[param]
        if do_unc:
            return x0, s0
        else:
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
                        0.5*self.gamma[self.required_params.index ( param ) ]*(np.sum(np.array(self.D1.dot(xa.T))**2))
                    der_cost[i:(i+self.n_elems)] = np.array( \
                        self.gamma[self.required_params.index ( param ) ]*np.dot((self.D1).T, \
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
        
        h = sp.lil_matrix ( (n ,n ) )
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
        nrows, ncols = self.nx # Needs checking...
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
                    
                    
                    d1 = np.ones(nrows*ncols, dtype=np.int8)*2
                    d1[::nrows] = 1
                    d1 [(nrows-1)::nrows] = 1

                    # The +1 or -1 diagonals are
                    d2 = -np.ones(nrows*ncols, dtype=np.int8)
                    d2[(nrows-1)::nrows] = 0

                    #DYsyn = np.diag(d1,k=0) + np.diag(d2[:-1],k=1) + np.diag(d2[:-1],k=-1)
                    #DYsparse = sp.dia_matrix (DYsyn, dtype=np.float32)
                    DYsparse = sp.dia_matrix ( (d1,0), shape=(nrows*ncols, nrows*ncols)) + \
                        sp.dia_matrix ( (np.r_[0,d2], 1), shape=(nrows*ncols, nrows*ncols)) + \
                        sp.dia_matrix ( (np.r_[d2, 0], -1), shape=(nrows*ncols, nrows*ncols))

                    d1 = 2*np.ones(nrows*ncols, dtype=np.int8) 
                    d1[:ncols] = 1
                    d1[-ncols:] = 1

                    d2 = -1*np.ones(nrows*ncols, dtype=np.int8)
                    DXsparse = sp.spdiags( [ d1, d2, d2], [0,ncols, -ncols], nrows*ncols, ncols*nrows)
                    
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
            self.n_bands, self.n_obs = self.observations.shape
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
                der_cost[j] = the_derivatives[i, : ].sum()
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
                dummy, df_0, dummy_fwd = self.calc_mismatch ( this_obsop, \
                    xs, this_obs, self.bu, *this_extra )
                iloc = 0
                iiloc = 0
                for i,fin_diff in enumerate(param_pattern):
                    if fin_diff == 1: # FIXED
                        continue                    
                    xxs = xs[i]*1
                    xs[i] += epsilon
                    dummy, df_1, dummy_fwd = self.calc_mismatch ( this_obsop, \
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

    def first_guess ( self, state_config, do_unc=False ):
        """
        A method to provide a first guess of the state. The idea here is to take the GPs, 
        and recast them, so that rather than provide an emulator, they provide a regressor
        from input reflectance/radiance etc to surface parameters
        """
        
        gps = create_inverse_emulators ( self.original_emulators, \
            self.band_pass, state_config )

        if do_unc:
            s0 = dict()
        x0 = dict()        
        for param, gp in gps.iteritems():
            x0[param] = np.zeros_like( self.observations[0,:, :].flatten())
            if do_unc:
                s0[param] = np.zeros_like( self.observations[0,:, :].flatten())
            xsol = gp.predict( self.observations[:, self.mask].T, do_unc=False )
            x0[param][self.mask.flatten()] = xsol[0]
            if do_unc:
                # for the pixels where we have observations, we can use the GP
                # uncertainty directly. For the one where we don't, we can use
                # the maximum uncertainty. Maybe this is even too confident!
                s0[param][self.mask.faltten()] = xsol[1]
                s0[param][~self.mask.flatten()] = s0[param][self.mask.flatten()].max()
            x0[param][~self.mask.flatten()] = x0[param][self.mask.flatten()].mean()
        if do_unc:
            return x0, s0
        else:
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
            fwd_model, partial_derv = \
                self.emulators[band].predict ( x_params[:, :].T, do_unc=False )
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
	# Define a 2D array same size as the_derivatives to store the main diagonal
	# Hessian (linear approximation term)
	diag_hessian = np.zeros_like ( the_derivatives )
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
        self.obs_op_grad = []
        
        for band in xrange ( self.n_bands ):
            # Run the emulator forward. Doing it for all pixels, or only for
            # the unmasked ones
            # Also, need to work out whether the size of the state is 
            # different to that of the observations (ie integrate over coarse res data)
            fwd_model,  partial_derv = \
                self.emulators[band].predict ( x_params[:, zmask.flatten()].T, do_unc=False)
            self.obs_op_grad.append ( partial_derv )
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
	      
	    diag_hessian[:, zmask.flatten()] += ( partial_derv**2/self.bu[band]**2 ).T
        
        j = 0
        self.diag_hess_vect = np.zeros_like ( der_cost )
        for  i, (param, typo) in enumerate ( state_config.iteritems()) :
            if typo == CONSTANT:
                der_cost[j] = the_derivatives[i, :].sum()
                self.diag_hess_vect[j] = diag_hessian[i, :].sum()
                j += 1
            elif typo == VARIABLE:
                n_elems = x_dict[param].size
                der_cost[j:(j+n_elems) ] = the_derivatives[i, :]
                self.diag_hess_vect[j:(j+n_elems)] = diag_hessian[i, :]
                j += n_elems
        self.gradient = der_cost # Store the gradient, we might need it later
        return cost, der_cost
    
    def der_der_cost ( self, x, state_config, state, epsilon=1.0e-5 ):
        """Numerical approximation to the Hessian"""
           
        """
        J''(x) = H'(x)^{T}C_{obs}^{T}H'(x) - H''(x)^{T}C_{obs}^{-1}(R - H(x))

        So the Hessian is made up of two terms: a first term which is the typically
        calculated linear approximation, made up of the gradient/jacobian, and a
        second term that we can see as a correction for non-linearities. This  
        second term requires the Hessian of the observation operator, as well as
        the misfit of observations and forward model

        
        The first term (linear approximation)
        ----------------------------------------------------
        
        This term is given by H'CH. Annoyingly, in Python, if H is a vector, and
        C is a matrix, this can be achieved by (H*C).T*H (it returns a matrix).
        In sparse terms, only C is sparse here (typically diagonal, or not too
        far off), so we have (H.dot(C)).T.dot(H)
        
        The obvious complications are to do with making sure the elements of H
        and C are consistent.
        
        1.- Run der_cost(x ) to make sure self.gradient is current and relates to 
        x
        2.- 
        
        The second term
        ----------------------------------
        
        This second is slightly more complicated. We need H'', the actual Hessian
        of the ObsOp. This we get from the emulators directly, but note that the
        emulator method returns **ALL** the input parameters (ie if you are only
        bothered about LAI, you still get the Hessian terms for all the other~10
        parameters), so some cleaning up of H'' is required. Further, we expect
        H'' to be sparse (only non-zero elements in the locations where 
        observations are available). 

        I have a problem visualising how this part of the equation works: if H''
        is a matrix and C_obs is a matrix, then the second term is a vector (the
        residuals is just a vector transformed by the product of two matrices). 
        Clearly, it must be a matrix, otherwise the sum in the original equation
        does not make sense.
        """
        N = x.size
        
        
        x_dict = state._unpack_to_dict ( x, do_invtransform=True )

        cost, cost_der = self.der_cost ( x_dict, state_config )
        main_diag = self.diag_hess_vect
        h = sp.dia_matrix (( self.diag_hess_vect, 0 ), shape=(N, N)).tolil()
        return h
        
        """
        
        N = x.size
        
        h = sp.lil_matrix ( ( N, N ), dtype=np.float32 )
        x_dict = state._unpack_to_dict ( x, do_invtransform=True )
        
        # We need the gradient of the observation operator. This is stored as
        # a list in the object, convert it to an array
        jacobian = np.array ( self.obs_op_grad )
        # The jacobian is Nbands, Nx*Ny, Nparams
        ## Here we need to do the linear bit of the matrix
        ## this means re-arranging the jacobian per observation
        
        i_unmasked = 0
        n_grid = self.nx*self.ny # The grid size
        n_obs = self.mask.flatten().sum() # The number of grid points with
                                          # observations availabl
        ###TODO define select pars, the selected parameters
        ###TODO Need to cope with CONSTANT parameters too
        # Now, loop over all pixels...
        n_pars = jacobian.shape[-1] # Number of parameters
        # Location of the parameters we'll be using. Note that this includes
        # both CONSTANT and VARIABLE parameters!
        sel_pars = [ i \
            for ( i, (k,v)) in enumerate ( state_config.iteritems() ) if v > 1 ]
        sel_pars = np.array ( sel_pars )
        for ipxl in xrange ( n_grid  ):
            if self.mask.flatten()[ipxl]:
                # This pixel isn't masked out...
                # Start by just selecting the jacobian parameters
                #############################################################
                ## THIS IS OVERCOMPLICATED, NEED TO CALL THE GP DIRECTLY
                ## and use that information here
                #############################################################
                Hs = np.zeros ( ( len(sel_pars), len(sel_pars) ))
                for b in xrange ( self.n_bands):
                    # Need to check that jacobian is Nbands, Nx*Ny, Npars
                    jac = jacobian [ b, ipxl, sel_pars] 
                        
                #############################################################
                ## The different resolution effect needs to be taken here
                #############################################################
                    #Hs = np.matrix ( jac ).T * np.matrix( \
                        #np.diag(np.ones(len(sel_pars))/(self.bu[b]**2))) * \
                        #np.matrix(jac)
                    C = np.matrix( \
                        np.diag(np.ones(len(sel_pars))/(self.bu[b]**2))) 
                    Hs += (np.matrix(jac).T*np.matrix(jac))*C

                # At this point, we have calcualted H'(x)C^{-1}H(x) for this
                # location in the grid, and need to place it in the right
                # place in the output Hessian
                for i in sel_pars:
                    for j in sel_pars: 
                        h [ i*n_grid + ipxl, j*n_grid + ipxl ] = Hs [ i, j ]
        # So this is the first approximation to the Hessian, bar the above todos
        return h # linear_approx
        
        
        
        
        
        #### This next bit is the calculation of (R-H(x))
        ##err = np.zeros_like ( self.observations )
        ##for band in xrange ( self.n_bands ):
            ##err[band, self.mask] = (  self.observations[band, self.mask] - \
                ##self.fwd_modelled_obs[band, self.mask] )
            
       
        
        ##########################################################################
        #### Numerical hessian code stuff                                       ##
        ##########################################################################
        ##df_0 = self.der_cost ( x_dict, state_config )[1]
        ##for i in xrange(N):
            ##xx0 = 1.*x[i]
            ##x[i] = xx0 + epsilon
            ##x_dict = state._unpack_to_dict ( x, do_invtransform=True )
            ##df_1 = self.der_cost ( x_dict, state_config )[1]
            ##h[i,:] = (df_1 - df_0)/epsilon
            ##x[i] = xx0
        ##return sp.lil_matrix ( h )
"""