#!/usr/bin/env python
"""An example script to demonstrate the smoothing of a univariate time series with gaps
"""
import numpy as np
from eoldas_ng import *

def create_data ( sigma_obs ):
    x = np.arange(1,366 )
    y_smooth = (1-np.cos(np.pi*x/366.)**2)
    y_noisy = y_smooth + np.random.randn(365)*sigma_obs
    return ( x, y_smooth, y_noisy )


if __name__ == "__main__":
    # Set prior
    # Set smoothness
    # Set observation
    # Set state
    # optimise
    sigma_obs = 0.15
    x, y_smooth, y_noisy = create_data ( sigma_obs )
    obs_mask = np.random.rand(x.shape[0])
    obs_mask = np.where ( obs_mask > 0.7, True, False )
    x_prior = np.ones_like(x)*0.5
    sigma_prior = np.array([1.0])
    gamma = 5000
    
    state_config = { 'magnitude': VARIABLE }
    state_grid = x
    default_values = { 'magnitude': 0.5 } # Unused
    the_state = State ( state_config, state_grid, default_values )
    the_smoother = TemporalSmoother ( state_grid, order=1, gamma=gamma )
    
    prior_mu = { 'magnitude': x_prior }
    prior_inv_cov = {'magnitude': (1./(sigma_prior**2)) }
    the_prior = Prior ( prior_mu, prior_inv_cov )
    
    the_observations = ObservationOperator ( y_noisy, sigma_obs, obs_mask)
    
    the_state.add_operator ( "Regularisation", the_smoother )
    the_state.add_operator ( "Prior", the_prior )
    the_state.add_operator ( "Observations", the_observations )
    retval = the_state.optimize( np.random.rand(365) )