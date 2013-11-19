#!/usr/bin/env python
from collections import OrderedDict

import numpy as np 


from operators import *
from create_emulators import *
from gp_emulator import MultivariateEmulator

if __name__ == "__main__":
    FIXED = 1
    CONSTANT = 2
    VARIABLE = 3
    # Create the state
    # First, define the state configuration dictionary
    
    state_config = OrderedDict ()
    
    #####state_config['n'] = CONSTANT
    #####state_config['cab'] = VARIABLE
    #####state_config['car'] = CONSTANT
    #####state_config['cbrown'] = VARIABLE
    #####state_config['cw'] = VARIABLE
    #####state_config['cm'] = VARIABLE
    #####state_config['lai'] = VARIABLE
    #####state_config['ala'] = CONSTANT
    ######state_config['lidfb'] = FIXED
    #####state_config['bsoil'] = CONSTANT
    #####state_config['psoil'] = VARIABLE
    ######state_config['hspot'] = CONSTANT

    state_config['n'] = FIXED
    state_config['cab'] = VARIABLE
    state_config['car'] = FIXED
    state_config['cbrown'] = FIXED
    state_config['cw'] = VARIABLE
    state_config['cm'] = FIXED
    state_config['lai'] = VARIABLE
    state_config['ala'] = FIXED
    #state_config['lidfb'] = FIXED
    state_config['bsoil'] = VARIABLE
    state_config['psoil'] = FIXED
    #state_config['hspot'] = CONSTANT
    
    
    
    
    # Now define the default values
    default_par = OrderedDict ()
    default_par['n'] = 1.5
    default_par['cab'] = 40.
    default_par['car'] = 10.
    default_par['cbrown'] = 0.01
    default_par['cw'] = 0.018 # Say?
    default_par['cm'] = 0.0065 # Say?
    default_par['lai'] = 2
    default_par['ala'] = 45.
    #default_par['lidfb'] = 0.
    default_par['bsoil'] = 1.
    default_par['psoil'] = 0.1
    #default_par['hspot'] = 0.01
    
    
    # Define boundaries
    #parameter_names = [ 'bsoil', 'cbrown', 'hspot', 'n', \
    #    'psoil', 'ala', 'lidfb', 'cab', 'car', 'cm', 'cw', 'lai' ]
    parameter_min = OrderedDict()
    parameter_max = OrderedDict()
    
    min_vals = [ 0.8, 0.2, 0.0, 0.0, 0.0043, 0.0017,0.01, 40, 0., 0., 0.]
    max_vals = [2.5, 77., 5., 1., 0.0713, 0.0331, 8., 50., 2., 1.]


        
        
    for i, param in enumerate ( state_config.keys() ):
        parameter_min[param] = min_vals[i]
        parameter_max[param] = max_vals[i]
    # Define the state grid. In time in this case
    state_grid = np.arange ( 1, 366 )
    # Define parameter transformations
    transformations = {
        'lai': lambda x: np.exp ( -x/2. ), \
        'cab': lambda x: np.exp ( -x/100. ), \
        'car': lambda x: np.exp ( -x/100. ), \
        'cw': lambda x: np.exp ( -50.*x ), \
        'cm': lambda x: np.exp ( -100.*x ), \
        'ala': lambda x: x/90. }
    inv_transformations = {
        'lai': lambda x: -2*np.log ( x ), \
        'cab': lambda x: -100*np.log ( x ), \
        'car': lambda x: -100*np.log( x ), \
        'cw': lambda x: (-1/50.)*np.log ( x ), \
        'cm': lambda x: (-1/100.)*np.log ( x ), \
        'ala': lambda x: 90.*x }
    
    # Define the state
    # L'etat, c'est moi
    state = State ( state_config, state_grid, default_par, \
        parameter_min, parameter_max )
    # Set the transformations
    state.set_transformations ( transformations, inv_transformations )
    

    mu_prior = OrderedDict ()
    prior_inv_cov = OrderedDict ()
    for param in state.parameter_min.iterkeys():
        mu_prior[param] = np.array([default_par[param]])
        prior_inv_cov[param] = np.array([1])
            #parameter_max[param] - parameter_min[param]*0.4)
        prior = Prior ( mu_prior, prior_inv_cov )
    
    x_dict = {}
    for param in state.parameter_min.iterkeys():
        if state_config[param] == CONSTANT:
            x_dict[param] = 0.5*default_par[param]
        elif state_config[param] == VARIABLE:
            #x_dict[param] = np.random.rand(365)*(parameter_max[param] - parameter_min[param]) + parameter_min[param]
            x_dict[param] = np.ones(365)*default_par[param]
        elif state_config[param] == FIXED:
            x_dict[param] = default_par[param]
            
    
    
    def cab_traj ( t ):
        out = t*0.
        w = np.where( t <= 0.5 )[0]
        out[w] = 10.5 + 208.7*t[w]
        w = np.where(t > 0.5)[0]
        out[w] = 219.2 - 208.7*t[w]
        return out

    trajectories = {
        'lai': lambda t: (0.21 + 3.51 * (np.sin(np.pi*t)**5)), 
        'cab': cab_traj,
        'cw': lambda t: (0.068/5 + 0.01*np.sin(np.pi * t+0.1) *  \
                 np.sin(6*np.pi*t + 0.1) ),
        'bsoil': lambda t: (2.5*(0.2 + 0.18*np.sin(np.pi*t) * \
                  np.sin(6*np.pi*t)) )
        }
    
    

    # Now, create some simulated data...

    parameter_grid = create_parameter_trajectories ( state, trajectories )

    # Now forward model the observations...
    doys, vza, sza, raa, rho = create_observations ( state, parameter_grid, \
            42, -8.)




    fnames = [ "/tmp/vza_30_sza_0_raa_40" ]
    # This comes from discretising vza & sza...
    szax = np.arange ( 0, 35, 5 )
    vzax = np.arange ( 0, 20, 5 )
    angles = [a  for a in itertools.product ( szax, vzax, [0] ) ]
    angles = [ [ 15, 15, 0] ]
#    emulators,  samples, validate = create_emulators ( state, fnames, angles=angles )   
#    for i,(s,v,r) in enumerate(angles):     
#        fname = "%02d_sza_%02d_vza_000_raa" % (s,v)
#        emulators[i].dump_emulator(fname)
    emulators = {}
    for i,(s,v,r) in enumerate(angles):     
        fname = "%02d_sza_%02d_vza_000_raa.npz" % (s,v)
        emulators[(v,s)]= MultivariateEmulator ( dump=fname )
        

    
    rho_big = np.zeros(( 7,365))
    mask = np.zeros(( 365, 4))
    time_grid = np.arange ( 1, 366 )

    for i in time_grid:
        if i in doys:
            rho_big[:, i] = rho[:, doys==i].squeeze()
            mask[ i, :] = [ 1, vza[doys==i], sza[doys==i],  raa[doys==i] ]

    bu = np.array( [ 0.00463576,  0.00540698,  0.004, 0.00431066,  \
        0.00678507,  0.00822998,  0.01])
    
    b_min = np.array( [ 620., 841, 459, 545, 1230, 1628, 2105] )
    b_max = np.array( [ 670., 876, 479, 565, 1250, 1652, 2155] )

    wv = np.arange ( 400, 2501 )
    band_pass = np.zeros((7,2101), dtype=np.bool)
    n_bands = b_min.shape[0]
    bw = np.zeros( n_bands )
    bh = np.zeros( n_bands )
    for i in xrange( n_bands ):
        band_pass[i,:] = np.logical_and ( wv >= b_min[i], \
                wv <= b_max[i] )
        bw[i] = b_max[i] - b_min[i]
        bh[i] = ( b_max[i] + b_min[i] )/2.


    #idoy = doys[0]
    #rhos = np.zeros((1, 365))
    #M = np.zeros((365, 4))
    #M[idoy, :] = mask[idoy,:]
    #rhos[ :, idoy] = rho_big[1, idoy]
    #bu = np.array(  [bu[1]] )
    #band_pass = np.atleast_2d ( [ band_pass[1] ] )
    #bw = np.atleast_2d([bw[1]])
    obs = ObservationOperatorTimeSeriesGP( time_grid, state, rho_big, mask, emulators, bu, band_pass, bw )

    temporal = TemporalSmoother ( time_grid, 1000, required_params=["lai"] )
    state.add_operator ( "Observations", obs )     
    state.add_operator ( "Prior", prior )
    state.add_operator ( "Smoother", temporal )
    
    #   
                
    x_dict = {}
    for i,k in enumerate(state.parameter_max.keys()):
        if state.state_config[k] == VARIABLE:
            x_dict[k] =  ( parameter_grid[i, :] )* + mu_prior[k]
        else:
            x_dict[k] = parameter_grid[i, 0]#*0+ mu_prior[k]
    
    X = state.pack_from_dict ( x_dict )
    x_dict['lai'] = np.ones(365)*mu_prior['lai']
    bounds = []
    for i in xrange(365):
        bounds.append ( state.bounds[6] )
        
    retval = state.optimize ( x_dict, bounds=bounds )
        
        