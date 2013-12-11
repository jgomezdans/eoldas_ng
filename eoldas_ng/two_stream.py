#!/usr/bin/env python
"""
The 2stream observation operator
"""

from collections import OrderedDict
import numpy as np


def two_stream_model ( x, sun_angle, structure_factor_zeta = 1., \
    structure_factor_zetastar = 1. ):
    """This function calculates absorption in the visible and NIR
    for a given parameter vector x.

    The parameter vector is:
    1. $\omega_{leaf,VIS}$
    2. $d_{leaf, VIS}$
    3. $r_{soil, VIS}$
    4. $\omega_{leaf,NIR}$
    5. $d_{leaf, NIR}$
    6. $r_{soil, NIR}$
    7. $LAI$
    
    """
    # TODO UCL-ism!!!!
    import sys
    sys.path.append ( "../../TwoStream" )
    from TwoSInterface import twostream_solver
    # These structural effective parameters are hardwired to be 1
    # Calculate leaf properties in VIS and NIR

    # Or, according to the paper...
    ####################### Pinty et al, 2008 VERSION ###########################################
    tvis = x[0]/(1.+x[1])
    rvis = x[1]*x[0]/(1+x[1])

    tnir = x[3]/(1.+x[4])
    rnir = x[4]*x[3]/(1+x[4])


    # Model visible
    collim_alb_tot_vis, collim_tran_tot_vis, collim_abs_tot_vis, \
        iso_alb_tot_vis, iso_tran_tot_vis, iso_abs_tot_vis = \
        twostream_solver( rvis, tvis, x[2], x[6], \
        structure_factor_zeta, structure_factor_zetastar, \
        sun_angle )
    # Model NIR
    collim_alb_tot_nir, collim_tran_tot_nir, collim_abs_tot_nir, \
        iso_alb_tot_nir, iso_tran_tot_nir, iso_abs_tot_nir = \
        twostream_solver( rnir, tnir, x[5], x[6], \
        structure_factor_zeta, structure_factor_zetastar, \
        sun_angle )
    # For fapar we return 
    #[ iso_abs_tot_vis, iso_abs_tot_nir]
    return  [ collim_alb_tot_vis, collim_alb_tot_nir ]

def create_emulators ( sun_angles, x_min, x_max, n_train=250, n_validate=1000 ):
    """This is a helper function to create emulators from the 2stream model.
    The emulators operate on all parameters (7) and the user needs to provide
    a numer of `sun_angles`. This then allows the inversion of BHR data from e.g.
    MODIS MCD43. We need to specify minimum and maximum boundaries for the 
    parameters (`x_min`, `x_max`). The number of training samples (`n_train`) 
    is set to 250 as this results in a very good emulation while still 
    being reasonably fast. The validation on 1000 randomly drawn parameters
    will be reported too. The output will be one dictionary, indexed by
    sun angle, one for VIS and one for NIR.
    
    """
    
    from gp_emulator import GaussianProcess, lhd
    import scipy.stats as ss

    n_params = x_min.size
    ## Putting boundaries on parameter space is useful
    #x_min = np.array ( 7*[ 0.001,] )
    #x_max = np.array ( 7*[0.95, ] )
    #x_max[-1] = 10.
    #x_min[1] = 0.
    #x_max[1] = 5.
    #x_min[4] = 0.
    #x_max[4] = 5.
    # First we create the sampling space for the emulators. In the
    # absence of any further information, we assume a uniform 
    # distribution between x_min and (x_max - x_min):
    dist = []
    for k in xrange( n_params ):
        dist.append ( ss.uniform ( loc=x_min[k], \
                              scale = x_max[k] - x_min[k] ) )
    # The training dataset is obtaiend by a LatinHypercube Design
    x_train = lhd(dist=dist, size=n_train )
    # The validation dataset is randomly drawn from within the 
    # parameter boundaries
    x_validate = np.random.rand (  n_validate, n_params  )*(x_max - x_min) + \
        x_min
    emu_vis = {}
    emu_nir = {}
    # We next loop over the input sun angles
    for sun_angle in sun_angles:
        # If we've done this sun angle before, skip it
        if not emu_vis.has_key ( sun_angle ):
            albedo_train = []
            albedo_validate = []
            # The following loop creates the validation dataset
            for i in xrange( n_validate ):
                [a_vis, a_nir] = two_stream_model ( x_validate[i,:], \
                    sun_angle )
                albedo_validate.append ( [a_vis, a_nir] )
            # The following loop creates the training dataset
            for i in xrange ( n_train ):
                [a_vis, a_nir] = two_stream_model ( x_train[i,:], \
                    sun_angle )
                albedo_train.append ( [a_vis, a_nir] )

            albedo_train = np.array ( albedo_train )
            albedo_validate = np.array ( albedo_validate )
            # The next few lines create and train the emulators
            # GP for visible
            gp_vis = GaussianProcess ( x_train, albedo_train[:,0])
            theta = gp_vis.learn_hyperparameters(n_tries=4)
            
            # GP for NIR
            gp_nir = GaussianProcess ( x_train, albedo_train[:,1])
            theta = gp_nir.learn_hyperparameters(n_tries=4)
            pred_mu, pred_var, par_dev = gp_vis.predict ( x_validate )
            r_vis = (albedo_validate[:,0] - pred_mu)
            pred_mu, pred_var, par_dev = gp_nir.predict ( x_validate )
            r_nir = (albedo_validate[:,1] - pred_mu)
            # Report some goodness of fit. Could do with more
            # stats, but for the time being, this is enough.
            print "Sun Angle: %g, RMSE VIS: %g, RMSE NIR: %g" % \
                ( sun_angle, r_vis.std(), r_nir.std() )
            emu_vis[sun_angle] = gp_vis
            emu_nir[sun_angle] = gp_nir
    emulators = {}
    for sun_angle in emu_vis.iterkeys():
        emulators[sun_angle] = [ emu_vis[sun_angle], emu_nir[sun_angle] ]
    return emulators
        
def select_emulator ( emulators, mask, itime ):

    sun_angle = mask[itime,1]
    vis = emulators[sun_angle][0].predict
    nir = emulators[sun_angle][1].predict
    return vis, nir
    
class ObservationOperatorTwoStream ( object ):
    """A GP-based observation operator"""
    def __init__ ( self, state_grid, state, observations, mask, emulators, bu ):
        """
         observations is an array with n_bands, nt observations. nt has to be the 
         same size as state_grid (can have dummny numbers in). mask is nt*4 
         (mask, vza, sza, raa) array.
         
         
        """
        self.state = state
        
        self.observations = observations
        
        self.mask = mask
    
        self.state_grid = state_grid
        self.emulators = emulators
        self.bu = bu
        
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
                x_params[ j, : ] = x_dict[param]
                
            elif typo == VARIABLE:
                x_params[ j, : ] = x_dict[param]

            j += 1
        

        for itime, tstep in enumerate ( self.state_grid ):
            if self.mask[itime, 0] == 0:
                # No obs here
                continue
            # We use the `get_emulator` method to select the required
            # emulator for this geometry, spectral setting etc
            obs_ops = self.get_emulator ( itime, self.mask, self.emulators )
            sigma_obs_vis, sigma_obs_vis = self.bu[ :, itime ]
            # forward model the proposal
            x = x_params[:, itime]
            model_albedo_vis, vis_var, vis_der = \
                obs_ops[0] ( np.atleast_2d(x) )
            model_albedo_nir, nir_var, nir_der = \
                obs_ops[1] ( np.atleast_2d(x) )
            # Calculate the actual cost
            this_cost = 0.5*( model_albedo_vis - albedo_vis )**2/sigma_obs_vis**2 + \
              0.5*( model_albedo_nir - albedo_nir  )**2/sigma_obs_nir**2
    
            # The partial derivatives of the cost function are then
            this_der= (1./sigma_obs_vis**2)*( model_albedo_vis - \
                albedo_vis )*vis_der + \
                (1./sigma_obs_nir**2)*( model_albedo_nir - albedo_nir )*nir_der 
            

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
        post_cov = np.linalg.inv (h)
        post_sigma = np.sqrt ( post_cov.diagonal() )
        return h, post_cov, post_sigma

        