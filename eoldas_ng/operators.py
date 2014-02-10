#!/usr/bin/env python
"""
EOLDAS ng
==========

A reorganisation of the EOLDAS codebase

"""

__author__  = "J Gomez-Dans"
__version__ = "1.0 (1.12.2013)"
__email__   = "j.gomez-dans@ucl.ac.uk"

import numpy as np
import scipy.optimize
from collections import OrderedDict


FIXED = 1
CONSTANT = 2
VARIABLE = 3


class AttributeDict(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    

def test_fwd_model ( x, the_emu, obs, bu, band_pass, bw):
    f,g = fwd_model ( the_emu, x, obs, bu, band_pass, bw )
    return f#,g.sum(axis=0)

def der_test_fwd_model ( x, the_emu, obs, bu, band_pass, bw):
    f,g = fwd_model ( the_emu, x, obs, bu, band_pass, bw )
    return g


def fwd_model ( gp, x, R, band_unc, band_pass, bw ):
        f, g = gp.predict ( np.atleast_2d( x ) )
        cost = 0
        der_cost = []
        
        for i in xrange( len(band_pass) ):
            d = f[band_pass[i]].sum()/bw[i] - R[i]
            #derivs = d*g[:, band_pass[i] ]/(bw[i]*(band_unc[i]**2))
            derivs = d*g[:, band_pass[i] ]/((band_unc[i]**2))
            cost += 0.5*np.sum(d*d)/(band_unc[i])**2
            der_cost.append ( np.array(derivs.sum( axis=1)).squeeze() )
        return cost, np.array( der_cost ).squeeze().sum(axis=0)

def downsample(myarr,factorx,factory):
    """
    Downsample a 2D array by averaging over *factor* pixels in each axis.
    Crops upper edge if the shape is not a multiple of factor.

    This code is pure numpy and should be fast.
    Leeched from <https://code.google.com/p/agpy/source/browse/trunk/agpy/downsample.py?r=114>
    """
    xs,ys = myarr.shape
    crarr = myarr[:xs-(xs % int(factorx)),:ys-(ys % int(factory))]
    dsarr = np.concatenate([[crarr[i::factorx,j::factory] 
        for i in range(factorx)] 
        for j in range(factory)]).mean(axis=0)
    return dsarr

def fit_smoothness (  x, sigma_model  ):
    """
    This function calculates the spatial smoothness constraint. We use
    a numpy strides trick to efficiently evaluate the neighbours. Note
    that there are no edges here, we just ignore the first and last
    rows and columns. Also note that we could have different weights
    for x and y smoothing (indeed, diagonal smoothing) quite simply
    """
    # Build up the 8-neighbours
    hood = np.array ( [  x[:-2, :-2], x[:-2, 1:-1], x[ :-2, 2: ], \
                    x[ 1:-1,:-2], x[1:-1, 2:], \
                    x[ 2:,:-2], x[ 2:, 1:-1], x[ 2:, 2:] ] )
    j_model = 0
    der_j_model = x*0
    for i in [1,3,4,6]:#range(8):
        j_model = j_model + 0.5*np.sum ( ( hood[i,:,:] - \
	  x[1:-1,1:-1] )**2 )/sigma_model**2
        der_j_model[1:-1,1:-1] = der_j_model[1:-1,1:-1] - \
	  ( hood[i,:,:] - x[1:-1,1:-1] )/sigma_model**2
    return ( j_model, 2*der_j_model )

def fit_observations_gauss ( x, obs, sigma_obs, qa, factor=1 ):
    """
    A fit to the observations term. This function returns the likelihood

    
    """
    if factor == 1:
        der_j_obs = np.where ( qa == 1, (x - obs)/sigma_obs**2, 0 )
        j_obs = np.where ( qa  == 1, 0.5*(x - obs)**2/sigma_obs**2, 0 )
        j_obs = j_obs.sum()
    else:
        j_obs, der_j_obs = fit_obs_spat ( x, obs, sigma_obs, qa, factor )

    return ( j_obs, der_j_obs )

  
def fit_obs_spat ( x, obs, sigma_obs, qa, factor ):
    """
    A fit to the observations term. This function returns the likelihood

    
    """
    import scipy.ndimage.interpolation
    
    xa = downsample ( x, factor[0], factor[1] )
    
    assert ( xa.shape == obs.shape )
    der_j_obs = np.where ( qa == 1, (xa - obs)/sigma_obs**2, 0 )
    
    der_j_obs = scipy.ndimage.interpolation.zoom ( der_j_obs, factor, order=1 )
    j_obs = np.where ( qa == 1, 0.5*(xa - obs)**2/sigma_obs**2, 0 )
    j_obs = j_obs.sum()
    return ( j_obs, der_j_obs )


         
##################################################################################        
##################################################################################        

class Prior ( object ):
    """A gaussian prior class"""
    def __init__ ( self, prior_mu, prior_inv_cov ):
        """The prior constructor.
        
        We take a dictionary with means and inverse covariance structures. The elements
        of the dictionary can either be 1-element or several element arrays (so you 
        could have an estimate of LAI for each time step from climatology, or a
        single value). The inverse covariance (or precision) matrix is either a single
        value ($1/\sigma^{2}$), or a full matrix. If you pass a single value for a
        VARIABLE parameter, it will be converted into a diagonal matrix automatically"""
        self.mu = prior_mu
        self.inv_cov = prior_inv_cov
        
                    
    
    def der_cost ( self, x_dict, state_config ):
        """Calculate the cost function and its partial derivatives for the prior object
        
        Takes a parameter dictionary, and a state configuration dictionary"""
        
        i = 0
        cost = 0
        
        n = 0
        for param, typo in state_config.iteritems():
            if typo == CONSTANT:
                n += 1
            elif typo == VARIABLE:
                n_elems = x_dict[param].size
                n += n_elems
        der_cost = np.zeros ( n )
        
        for param, typo in state_config.iteritems():
            
            if typo == FIXED: # Default value for all times
                # Doesn't do anything so we just skip
                pass
                
            elif typo == CONSTANT: # Constant value for all times
                
                cost = cost + 0.5*( x_dict[param] - self.mu[param])**2*self.inv_cov[param]
                der_cost[i] = ( x_dict[param] - self.mu[param])*self.inv_cov[param]
                
                i += 1                
                
            elif typo == VARIABLE:
                
                if self.inv_cov[param].size == 1:
                    sigma = self.inv_cov[param]
                    self.inv_cov[param] = np.diag( np.ones(n_elems)*sigma )
                    
                cost_m = ( x_dict[param].flatten() - self.mu[param]).dot ( \
                            self.inv_cov[param] )
                cost = cost + 0.5*(cost_m*(x_dict[param].flatten() - \
                            self.mu[param])).sum()
                der_cost[i:(i+n_elems)] = cost_m                                         
                
                i += n_elems
        
        return cost, der_cost
    

    def der_der_cost ( self, x, state_config ):
        # Hessian is just C_{prior}^{-1}
        # Needs some thinking for getting parameters in the
        # right positions etc
        pass
    
    


class TemporalSmoother ( object ):
    """A temporal smoother class"""
    def __init__ ( self, state_grid, gamma, order=1, required_params = None  ):
        self.order = order
        self.n_elems = state_grid.shape[0]
        I = np.identity( state_grid.shape[0] )
        self.D1 = np.matrix(I - np.roll(I,1))
        self.D1 = self.D1 * self.D1.T
        self.gamma = gamma
        self.required_params = required_params
        
    def der_cost ( self, x_dict, state_config):
        """Calculate the cost function and its partial derivs for a time smoother
        
        Takes a parameter dictionary, and a state configuration dictionary"""
        i = 0
        cost = 0
        n = 0
        self.required_params = self.required_params or state_config.keys()
        
        
        ## This is a nice idea if you wanted to e.g. solve for
        ## gamma....
        ##if x_dict.has_key ( 'gamma' ):
            ##self.gamma = x_dict['gamma']
            ##x_dict.pop ( 'gamma' )
        n = 0
        for param, typo in state_config.iteritems():
            if typo == CONSTANT:
                n += 1
            elif typo == VARIABLE:
                n_elems = x_dict[param].size
                n += n_elems
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
                    
                    xa = np.matrix ( x_dict[param] )
                    
                    cost = cost + \
                        0.5*self.gamma*(np.sum(np.array(self.D1.dot(xa.T))**2))
                    der_cost[i:(i+self.n_elems)] = np.array( self. gamma*np.dot((self.D1).T, \
                        self.D1*np.matrix(xa).T)).squeeze()
                    #der_cost[i] = 0
                    #der_cost[i+n_elems-1] = 0
                i += self.n_elems
                
                
        return cost, der_cost
    
    def der_der_cost ( self ):
        """The Hessian (rider)"""
        return self.gamma*np.dot ( self.D1,np.eye( self.n_elems )).dot( \
            self.D1.T)
            

class SpatialSmoother ( object ):
    """MRF prior"""
    def __init__ ( self, state_grid, gamma, required_params = None  ):
        self.nx = state_grid.shape
        self.gamma = gamma
        self.required_params = required_params
        
    def der_cost ( self, x_dict, state_config):
        """Calculate the cost function and its partial derivs for a time smoother
        
        Takes a parameter dictionary, and a state configuration dictionary"""
        i = 0
        cost = 0
        n = 0
        self.required_params = self.required_params or state_config.keys()
        

        n = 0
        for param, typo in state_config.iteritems():
            if typo == CONSTANT:
                n += 1
            elif typo == VARIABLE:
                n_elems = x_dict[param].size
                n += n_elems
            
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
                    
                    xa = x_dict[param].reshape( self.nx )
                    cost, dcost = fit_smoothness ( xa, self.gamma )
                    der_cost[i:(i+n_elems)] = dcost.flatten()
                i += n_elems
                
                
        return cost, der_cost
    def der_der_cost ( x, state_config ):
        # TODO Clear how it goes for single parameter, but for
        # multiparameter, it can easily get tricky. Also really
        # need to have all mxs in sparse format, otherwise, they
        # don't fit in memory.
        pass

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
        """Calculate the cost function and its partial derivs for identity obs op
        
        Takes a parameter dictionary, and a state configuration dictionary"""
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
    
    def der_der_cost ( self, x, state_config ):
        # Hessian is just C_{obs}^{-1}?
        hess_obs = np.zeros( x.size )
        hess_obs[ self.mask ] = (1./self.sigma_obs**2)
        hess_obs = np.diag( hess_obs )
        #TODO should be sparse!!!
        return hess_obs

        
class ObservationOperatorTimeSeriesGP ( object ):
    """A GP-based observation operator"""
    def __init__ ( self, state_grid, state, observations, mask, emulators, bu, band_pass=None, bw=None ):
        """
         observations is an array with n_bands, nt observations. nt has to be the 
         same size as state_grid (can have dummny numbers in). mask is nt*4 
         (mask, vza, sza, raa) array.
         
         
        """
        self.state = state
        self.observations = observations
        try:
            self.n_bands, self.nt = self.observations.shape
        except:
            raise ValueError, "Typically, obs should be n_bands * nt"
        self.mask = mask
        assert ( self.nt ) == mask.shape[0]
        self.state_grid = state_grid
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
        
        """
        i = 0
        cost = 0.
        n = 0
        n = 0
        for param, typo in state_config.iteritems():
            if typo == CONSTANT:
                n += 1
            elif typo == VARIABLE:
                n_elems = len ( x_dict[param] )
                n += n_elems
        der_cost = np.zeros ( n )
        x_params = np.empty ( ( len( x_dict.keys()), self.nt ) )
        j = 0
        ii = 0
        the_derivatives = np.zeros ( ( len( x_dict.keys()), self.nt ) )
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
        

        for itime, tstep in enumerate ( self.state_grid ):
            if self.mask[itime, 0] == 0:
                # No obs here
                continue
            # tag here is needed to look for the emulator for this geometry
            tag = tuple((5*(self.mask[itime, 1:3].astype(np.int)/5)).tolist())
            the_emu = self.emulators[ tag ]

            this_cost, this_der = fwd_model ( the_emu, x_params[:, itime], \
                 self.observations[:, itime], self.bu, self.band_pass, \
                 self.bw )
            
            #g = scipy.optimize.approx_fprime ( x_params[:, itime], test_fwd_model, 1e-10, the_emu, self.observations[:,itime], self.bu, self.band_pass, self.bw )
            cost += this_cost
            the_derivatives[ :, itime] = this_der
            
            
        j = 0
        for  i, (param, typo) in enumerate ( state_config.iteritems()) :
            if typo == CONSTANT:
                der_cost[j] = the_derivatives[i, 0]
                j += 1
            elif typo == VARIABLE:
                n_elems = len ( x_dict[param] )
                der_cost[j:(j+n_elems) ] = the_derivatives[i, :]
                j += n_elems
        
        return cost, der_cost
    def der_der_cost ( self, x, state_config, epsilon=1.0e-9 ):
        """Numerical approximation to the Hessian"""
            
        N = x.size
        h = np.zeros((N,N))
        df_0 = self.der_cost ( x, state_config )[1]
        for i in xrange(N):
            xx0 = 1.*x[i]
            x[i] = xx0 + epsilon
            df_1 = self.der_cost ( x, state_config )[1]
            h[i,:] = (df_1 - df_0)/epsilon
            x[i] = xx0
        return h

         
class ObservationOperatorImageGP ( object ):
    """A GP-based observation operator"""
    def __init__ ( self, state_grid, state, observations, mask, emulators, bu, band_pass=None, bw=None ):
        """
         observations is an array with n_bands, nt observations. nt has to be the 
         same size as state_grid (can have dummny numbers in). mask is nt*4 
         (mask, vza, sza, raa) array.
         
         
        """
        self.state = state
        self.observations = observations
        try:
            self.n_bands, self.nt = self.observations.shape
        except:
            raise ValueError, "Typically, obs should be n_bands * nt"
        self.mask = mask
        assert ( self.nt ) == mask.shape[0]
        self.state_grid = state_grid
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
        
        """
        i = 0
        cost = 0.
        n = 0
        n = 0
        for param, typo in state_config.iteritems():
            if typo == CONSTANT:
                n += 1
            elif typo == VARIABLE:
                n_elems = len ( x_dict[param] )
                n += n_elems
        der_cost = np.zeros ( n )
        x_params = np.empty ( ( len( x_dict.keys()), self.nt ) )
        j = 0
        ii = 0
        the_derivatives = np.zeros ( ( len( x_dict.keys()), self.nt ) )
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
        

        for itime, tstep in enumerate ( self.state_grid ):
            if self.mask[itime, 0] == 0:
                # No obs here
                continue
            # tag here is needed to look for the emulator for this geometry
            tag = tuple((5*(self.mask[itime, 1:3].astype(np.int)/5)).tolist())
            the_emu = self.emulators[ tag ]

            this_cost, this_der = fwd_model ( the_emu, x_params[:, itime], \
                 self.observations[:, itime], self.bu, self.band_pass, \
                 self.bw )
            
            #g = scipy.optimize.approx_fprime ( x_params[:, itime], test_fwd_model, 1e-10, the_emu, self.observations[:,itime], self.bu, self.band_pass, self.bw )
            cost += this_cost
            the_derivatives[ :, itime] = this_der
            
            
        j = 0
        for  i, (param, typo) in enumerate ( state_config.iteritems()) :
            if typo == CONSTANT:
                der_cost[j] = the_derivatives[i, 0]
                j += 1
            elif typo == VARIABLE:
                n_elems = len ( x_dict[param] )
                der_cost[j:(j+n_elems) ] = the_derivatives[i, :]
                j += n_elems
        
        return cost, der_cost
    def der_der_cost ( self, x, state_config, epsilon=1.0e-9 ):
        """Numerical approximation to the Hessian"""
            
        N = x.size
        h = np.zeros((N,N))
        df_0 = self.der_cost ( x, state_config )[1]
        for i in xrange(N):
            xx0 = 1.*x[i]
            x[i] = xx0 + epsilon
            df_1 = self.der_cost ( x, state_config )[1]
            h[i,:] = (df_1 - df_0)/epsilon
            x[i] = xx0
        return h
