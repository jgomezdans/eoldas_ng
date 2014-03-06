

def get_problem_size ( x_dict, state_config ):
    """This function reports
    1. The number of parameters `n_params`
    2. The size of the state

    Parameters
    -----------
    x_dict: dict
        An (ordered) dictionary with the state
    state_config: dict
        A state configuration ordered dictionary
    """
    n_params = 0
    for param, typo in state_config.iteritems():
        if typo == CONSTANT:
            n_params += 1
        elif typo == VARIABLE:
            n_elems = x_dict[param].size
            n_params += n_elems
    return n_params, n_elems

def test_fwd_model ( x, the_emu, obs, bu, band_pass, bw):
    f,g = fwd_model ( the_emu, x, obs, bu, band_pass, bw )
    return f#,g.sum(axis=0)

def der_test_fwd_model ( x, the_emu, obs, bu, band_pass, bw):
    f,g = fwd_model ( the_emu, x, obs, bu, band_pass, bw )
    return g


def fwd_model ( gp, x, R, band_unc, band_pass, bw ):
    """
    A generic forward model using GPs. We pass the gp object as `gp`,
    the value of the state as `x`, the observations as `R`, observational
    uncertainties per band as `band_unc`, and spectral properties of 
    bands as `band_pass` and `bw`.
    
    Returns
    -------
    
    The cost associated with x, and the partial derivatives.
    
    """
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
