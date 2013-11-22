from operators import *
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict



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



###rows = 100
###cols = 100
###sigma_obs = 0.15
###R, C, ndvi_true, ndvi_obs, qa_flag = create_data ( rows, cols )


###state_config = { 'magnitude': VARIABLE } # We solve for a parameter called 
###'magnitude'
###default_values = { 'magnitude': 0.5 } 
#### The default value is 0.5 but since it's defined
#### as VARIABLE above, this definition has no effect
###parameter_min = OrderedDict ()
###parameter_max = OrderedDict ()
###parameter_min [ 'magnitude' ] = 0.
###parameter_max [ 'magnitude' ] = 1.
###state_grid = np.arange ( 1, rows*cols + 1 ).reshape((rows, cols)) # A 2d grid
#### Now, just define the state
###the_state = State ( state_config, state_grid, default_values, parameter_min, \
                   ###parameter_max)


###x_dict = {'magnitude': np.ones(rows*cols)*0.25}
######################################################
###### The prior
##########################################################
###mu_prior = { 'magnitude': np.array([0.5]) }
###inv_cov_prior = { 'magnitude': np.array([1./(2*2)]) }
###the_prior = Prior ( mu_prior, inv_cov_prior ) # Uninformative prior

###the_state.add_operator ( "Prior", the_prior )

###the_smoother = SpatialSmoother ( state_grid, 0.1 )

###the_state.add_operator ( "Smoother", the_smoother )

###the_obs = ObservationOperator(state_grid, ndvi_obs, sigma_obs, qa_flag )

###the_state.add_operator ( "Observations", the_obs )




rows = 80
cols = 80
sigma_obs = 0.15
R, C, ndvi_true_hires, ndvi_obs_hires, qa_flag_hires = create_data ( rows, cols, obs_off=0.6, noise=0.20, window_size=11 )

cmap=plt.cm.spectral
cmap.set_bad('0.8')
plt.subplot ( 2,2,1 )
plt.imshow ( ndvi_true_hires, interpolation='nearest', vmin=0, vmax=1, cmap=cmap )
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.ylabel("High Resolution")
plt.title("'True'")
plt.subplot ( 2,2,2)
plt.title("Observed")
plt.imshow ( np.ma.array(ndvi_obs_hires,mask=qa_flag_hires==0), \
    interpolation='nearest', vmin=0, vmax=1, cmap=cmap )
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.ylabel("$\sigma=0.25$\nObsOff=60%%")
rows = 40
cols = 40
sigma_obs = 0.15
R, C, ndvi_true_lores, ndvi_obs_lores, qa_flag_lores = create_data ( rows, cols, noise=0.15,obs_off=0.3)

cmap=plt.cm.spectral
cmap.set_bad('0.8')
plt.subplot ( 2,2,3 )
plt.imshow ( ndvi_true_lores, interpolation='nearest', vmin=0, vmax=1, cmap=cmap )
plt.xticks(visible=False)
plt.yticks(visible=False)


plt.ylabel("Low Resolution")
plt.subplot ( 2,2,4)
plt.imshow ( np.ma.array(ndvi_obs_lores,mask=qa_flag_lores==0), \
    interpolation='nearest', vmin=0, vmax=1, cmap=cmap )
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.ylabel("$\sigma=0.15$\nObsOff=20%%")
cax = plt.axes([0.01, 0.05, 0.85, 0.05])
plt.colorbar(cax=cax,orientation='horizontal')


state_config = { 'magnitude': VARIABLE } # We solve for a parameter called 
'magnitude'
default_values = { 'magnitude': 0.5 } 
# The default value is 0.5 but since it's defined
# as VARIABLE above, this definition has no effect
parameter_min = OrderedDict ()
parameter_max = OrderedDict ()
parameter_min [ 'magnitude' ] = 0.
parameter_max [ 'magnitude' ] = 1.

rows = 80
cols = 80
x_dict = {'magnitude': np.ones(rows*cols)*0.25}
state_grid = np.arange ( 1, rows*cols + 1 ).reshape((rows, cols)) # A 2d grid
# Now, just define the state
the_state = State ( state_config, state_grid, default_values, parameter_min, \
                   parameter_max)

###################################################
### The prior
#######################################################
mu_prior = { 'magnitude': np.array([0.5]) }
inv_cov_prior = { 'magnitude': np.array([1./(2*2)]) }
the_prior = Prior ( mu_prior, inv_cov_prior ) # Uninformative prior

the_state.add_operator ( "Prior", the_prior )

the_smoother = SpatialSmoother ( state_grid, 0.1 )

the_state.add_operator ( "Smoother", the_smoother )

the_hires_obs = ObservationOperator(state_grid, ndvi_obs_hires, sigma_obs, qa_flag_hires )

the_state.add_operator ( "HiRes Observations", the_hires_obs )

the_lores_obs = ObservationOperator ( state_grid, ndvi_obs_lores, sigma_obs, qa_flag_lores, factor=[2,2])

the_state.add_operator ( "LoRes Observations", the_lores_obs )
