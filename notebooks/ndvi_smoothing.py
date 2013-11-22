import sys
sys.path.append("..")
from eoldas_ng import *
state_config = { 'magnitude': VARIABLE } # We solve for a parameter called 'magnitude'
default_values = { 'magnitude': 0.5 } # The default value is 0.5 but since it's defined
                                      # as VARIABLE above, this definition has no effect
parameter_min = OrderedDict ()
parameter_max = OrderedDict ()
parameter_min [ 'magnitude' ] = 0.
parameter_max [ 'magnitude' ] = 1.5
state_grid = np.arange ( 1, 366 ) # A daily time series with 1 day sampling
# Now, just define the state
the_state = State ( state_config, state_grid, default_values, parameter_min, \
                   parameter_max)

def create_data ( sigma_obs ):
    """Creates a 365-step long observations, contaminated with 
    additive Gaussian iid noise of standard deviation `sigma_obs`. 
    Random observations will be dropped by setting `obs_mask` to 
    `False`. We return the time axis, the original "clean" data, the
    noisy observations and the observation mask."""
    x = np.arange(1,366 )
    y_smooth = (1-np.cos(np.pi*x/366.)**2)
    y_noisy = y_smooth + np.random.randn(365)*sigma_obs
    obs_mask = np.random.rand(x.shape[0])
    obs_mask = np.where ( obs_mask > 0.7, True, False )

    return ( x, y_smooth, y_noisy, obs_mask )
sigma_obs = 0.15 # Observational uncertainty
( x, y_smooth, y_noisy, obs_mask ) = create_data ( sigma_obs )
plt.plot ( x, y_smooth, '-', lw=1.5, label="'True' data")
yN = np.ma.array ( y_noisy, mask=~obs_mask)

the_observations = ObservationOperator ( state_grid, y_noisy, sigma_obs, obs_mask.astype(np.int8))
x_dict = {'magnitude':np.ones(365)*0.5}
the_state.add_operator ( "Observations", the_observations )
mu_prior = { 'magnitude': np.array([0.5]) }
inv_cov_prior = { 'magnitude': np.array([1./(2*2)]) }
the_prior = Prior ( mu_prior, inv_cov_prior ) # Uninformative prior
the_model = TemporalSmoother ( state_grid, gamma=5000 )
the_state.add_operator ( "Prior", the_prior )
the_state.add_operator ( "Model", the_model )


def smothd ( x ):
    x_dict['magnitude'] = 1.*x
    f1,g1 = the_observations.der_cost ( x_dict, state_config )
    f2,g2 = the_prior.der_cost ( x_dict, state_config )
    f3,g3 = the_model.der_cost ( x_dict, state_config )

    return f1 + f2 + f3, g1 + g2 + g3

 