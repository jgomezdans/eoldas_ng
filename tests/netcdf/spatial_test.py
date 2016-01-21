import numpy as np
import matplotlib.pyplot as plt

from eoldas_ng_observations import OutputFile

from eoldas_ng import *
from collections import OrderedDict
import netCDF4


def gauss_kern(size, sizey=None):
    """ 
    Returns a normalized 2D gauss kernel array for convolutions 
    From http://www.scipy.org/Cookbook/SignalSmooth
    """
    import numpy as np
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()


def create_data ( nr,nc,n_per=1, noise=0.15, obs_off=0.1, \
                                        window_size=5):
    """
    Create synthetic "NDVI-like" data for a fictitious space series. We return
    the original data, noisy data (using IID Gaussian noise), the QA flag as well
    as the time axis.
    
    Missing observations are simulated by drawing a random number between 0 and 1
    and checking against obs_off.
    
    Parameters
    ----------
    nc : number of columns
    nr : number of rows

    n_per : integer
    Observation periodicity. By default, assumes every sample

    noise : float
    The noise standard deviation. By default, 0.15

    obs_off : float
    The threshold to decide on missing observations in the series.

    
    """
    import numpy as np
    from scipy import signal

    r,c = np.mgrid[1:nr+1:n_per,1:nc+1:n_per]
    ndvi_clean =  np.clip(np.sin(np.pi*2*(r-1)/nr)*np.sin(np.pi*2*(c-1)/nc), 0,1) 
    ndvi = ndvi_clean.copy()
    # add Gaussian noise of sd noise
    ndvi = np.random.normal(ndvi,noise,ndvi.shape)
     
    # set the qa flags for each sample to 1 (good data)
    qa_flag = np.ones_like ( ndvi).astype( np.int32 )
    passer = np.random.rand(ndvi.shape[0],ndvi.shape[1])
    if window_size >0:
        # force odd
        window_size = 2*(window_size/2)+1
        # smooth passer
        g = gauss_kern(window_size)
        passer = signal.convolve(passer,g,mode='same')
    # assign a proportion of the qa to 0 from an ordering of the smoothed 
    # random numbers
    qf = qa_flag.flatten()
    qf[np.argsort(passer,axis=None)[:passer.size * obs_off]]  = 0
    qa_flag = qf.reshape(passer.shape)
    return ( r,c , ndvi_clean, ndvi, qa_flag )


of = OutputFile ("spatial_example.nc", x=np.arange ( 1, 91), y = np.arange(1,91))
from eoldas_ng import MetaState
metadata = MetaState()
metadata.add_variable ( "magnitude","-","magnitude","mag")

rows = 90
cols = 90
sigma_obs = 0.15
R, C, ndvi_true_hires, ndvi_obs_hires, qa_flag_hires = create_data ( rows, cols, \
        obs_off=0.6, noise=0.25, window_size=11 )

rows = 30
cols = 30
R, C, ndvi_true_lores, ndvi_obs_lores, qa_flag_lores = create_data ( rows, cols, \
        noise=0.10, obs_off=0.4)

cmap=plt.cm.spectral
cmap.set_bad('0.8')



    
rows=90
cols=90
cmap=plt.cm.spectral
cmap.set_bad('0.8')
x_dict = {'magnitude': np.ones(rows*cols)*0.25}
state_config = { 'magnitude': VARIABLE } # We solve for a parameter called     'magnitude'
default_values = { 'magnitude': 0.5 } 
# The default value is 0.5 but since it's defined
# as VARIABLE above, this definition has no effect
parameter_min = OrderedDict ()
parameter_max = OrderedDict ()
parameter_min [ 'magnitude' ] = 0.
parameter_max [ 'magnitude' ] = 1.

x_dict = {'magnitude': np.ones(rows*cols)*0.25}
state_grid = np.arange ( 1, rows*cols + 1 ).reshape((rows, cols)) # A 2d grid
# Now, just define the state
the_state = State ( state_config, state_grid, default_values, parameter_min, \
                    parameter_max, output_name=of)
the_state.set_metadata ( metadata )
###################################################
### The prior
#######################################################
mu_prior = { 'magnitude': np.array([0.5]) }
inv_cov_prior = { 'magnitude': np.array([1./(2*2)]) }
the_prior = Prior ( mu_prior, inv_cov_prior ) # Uninformative prior

#the_state.add_operator ( "Prior", the_prior )

the_smoother = SpatialSmoother ( state_grid, 0.1 )

the_state.add_operator ( "Smoother", the_smoother )

the_hires_obs = ObservationOperator(state_grid, ndvi_obs_hires, 0.25, \
                                    qa_flag_hires )

the_state.add_operator ( "HiRes Observations", the_hires_obs )

the_lores_obs = ObservationOperator ( state_grid, ndvi_obs_lores, 0.1, \
                                        qa_flag_lores, factor=[3,3])

the_state.add_operator ( "LoRes Observations", the_lores_obs )

the_smoother.gamma = 0.04616#opt_gamma
retval = the_state.optimize ( x_dict, do_unc=True )

f = netCDF4.Dataset ( "spatial_example.nc")
for k in [ "real_map", "real_ci95pc", "real_ci75pc", "real_ci25pc", "real_ci5pc"]:
    in_file_data = f.groups[k].variables['magnitude'][:,:]
    print k, np.allclose ( retval[k]['magnitude'], in_file_data )
