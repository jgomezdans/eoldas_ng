import numpy as np
import netCDF4
import matplotlib.pyplot as plt

from eoldas_ng_observations import OutputFile

from eoldas_ng import *
from collections import OrderedDict

state_config = { 'magnitude': VARIABLE } # We solve for a parameter called 'magnitude'
default_values = { 'magnitude': 0.5 } # The default value is 0.5 but since it's defined
                                      # as VARIABLE above, this definition has no effect
parameter_min = OrderedDict ()
parameter_max = OrderedDict ()
parameter_min [ 'magnitude' ] = 0.
parameter_max [ 'magnitude' ] = 1.
state_grid = np.arange ( 1, 366 ) # A daily time series with 1 day sampling

# Now, just define the state
# Saving to netcdf like a bosss

of = OutputFile ("timeseries_eoldas_test.nc")
the_state = State ( state_config, state_grid, default_values, parameter_min, 
                   parameter_max, output_name=of )

def dbl_logistic_model ( p, x ):
    """A double logistic model, as in Sobrino and Juliean, or Zhang et al"""
    return p[0] + p[1]* ( 1./(1+np.exp(p[2]*(x-p[3]))) + 
                          1./(1+np.exp(-p[4]*(x-p[5])))  - 1 )
def create_data ( sigma_obs ):
    """Creates a 365-step long observations, contaminated with 
    additive Gaussian iid noise of standard deviation `sigma_obs`. 
    Random observations will be dropped by setting `obs_mask` to 
    `False`. We return the time axis, the original "clean" data, the
    noisy observations and the observation mask."""
    x = np.arange(1,366 )
    y_smooth = dbl_logistic_model ( [0.1, 0.85, -0.1, 60, -0.05, 240], x )
    y_noisy = y_smooth + np.random.randn(365)*sigma_obs
    obs_mask = np.random.rand(x.shape[0])
    obs_mask = np.where ( obs_mask > 0.94, True, False )

    return ( x, y_smooth, y_noisy, obs_mask )


sigma_obs = 0.15 # Observational uncertainty
( x, y_smooth, y_noisy, obs_mask ) = create_data ( sigma_obs )
yN = np.ma.array ( y_noisy, mask=~obs_mask)



metadata = MetaState()
metadata.add_variable ( "magnitude","-","magnitude","mag")
the_state.set_metadata ( metadata )
the_observations = ObservationOperator ( state_grid, y_noisy, sigma_obs, obs_mask)
the_state.add_operator ( "Observations", the_observations )
#######################################################
####   
#### Set the prior up!
####
#######################################################

mu_prior = { 'magnitude': np.array([0.5]) }
inv_cov_prior = { 'magnitude': np.array([1./(2*2)]) }
the_prior = Prior ( mu_prior, inv_cov_prior ) # Uninformative prior
#######################################################
####   
#### Set the model up!
####
#######################################################
the_model = TemporalSmoother ( state_grid, 5e3, required_params=["magnitude"] )

#######################################################
####   
#### Add the prior and the smoother to the state
#### Remember we had added the observations already before
####
#######################################################

the_state.add_operator ( "Prior", the_prior )
the_state.add_operator ( "Model", the_model )


retval = the_state.optimize( np.random.rand(y_smooth.size), do_unc=True )

f = netCDF4.Dataset ( "timeseries_eoldas_test.nc")
for k in [ "real_map", "real_ci95pc", "real_ci75pc", "real_ci25pc", "real_ci5pc"]:
    in_file_data = f.groups[k].variables['magnitude'][:,:]
    print k, np.allclose ( retval[k]['magnitude'], in_file_data )
