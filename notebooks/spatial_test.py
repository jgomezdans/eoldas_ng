#!/usr/bin/env python

import numpy as np
from collections import OrderedDict
try:
    from operators import *
except ImportError:
    import sys
    sys.path.append ( "../eoldas_ng/")
    from operators import *
from state import *
from create_emulators import *


def proba_spectral():
    # Selected bands in nm
    # proba_sel_bands = np.array ( [ 783, 665, 945, 705, 740, 443, 490, 560 ] )
    # Selected bands in band position according to bands_txt below:
    # proba_sel_bands = np.array([40, 23, 56, 29, 34,  1,  6, 13])
    # From http://www.uv.es/gcamps/papers/Verrelst10_livingplanet.pdf

    bands_txt = """410.501      405.633      415.244      9.61120
      441.286      436.253      446.572      10.3190
      451.155      446.572      455.890      9.31790
      460.776      455.890      465.796      9.90570
      470.956      465.796      476.259      10.4620
      481.712      476.259      487.298      11.0390
      491.899      487.298      496.596      9.29790
      501.417      496.596      506.357      9.76010
      511.431      506.357      516.657      10.3000
      522.043      516.657      527.573      10.9150
      531.848      527.573      536.238      8.66460
      542.221      536.238      548.447      12.2090
      553.251      548.447      558.182      9.73490
      563.245      558.182      568.430      10.2470
      573.769      568.430      579.258      10.8280
      582.975      579.258      586.800      7.54180
      592.636      586.800      598.645      11.8440
      604.822      598.645      611.168      12.5230
      615.512      611.168      619.917      8.74920
      624.419      619.917      628.969      9.05170
      633.636      628.969      638.395      9.42590
      643.218      638.395      648.135      9.73960
      653.172      648.135      658.276      10.1410
      663.488      658.276      668.815      10.5380
      674.242      668.815      679.748      10.9320
      682.559      679.748      685.324      5.57530
      688.174      685.324      691.070      5.74610
      693.963      691.070      696.883      5.81340
      699.860      696.883      702.841      5.95720
      705.839      702.841      708.888      6.04680
      711.954      708.888      715.058      6.17070
      718.192      715.058      721.320      6.26160
      724.529      721.320      727.724      6.40440
      730.932      727.724      734.266      6.54210
      737.563      734.266      740.901      6.63450
      744.231      740.901      747.686      6.78460
      751.100      747.686      754.582      6.89580
      758.096      754.582      761.634      7.05270
      765.172      761.634      768.737      7.10220
      772.388      768.737      776.076      7.33970
      779.735      776.076      783.491      7.41460
      787.287      783.491      791.046      7.55520
      794.949      791.046      798.750      7.70350
      802.702      798.750      806.614      7.86430
      810.582      806.614      814.631      8.01700
      831.065      822.750      839.468      16.7170
      843.759      839.468      848.038      8.57030
      852.458      848.038      856.874      8.83580
      861.241      856.874      865.688      8.81430
      870.174      865.688      874.757      9.06900
      879.316      874.757      883.899      9.14150
      888.505      883.899      893.240      9.34100
      898.019      893.240      902.751      9.51130
      907.533      902.751      912.337      9.58560
      917.255      912.337      922.147      9.81040
      927.173      922.147      932.027      9.88020
      942.121      932.027      952.277      20.2490
      957.371      952.277      962.585      10.3070
      967.744      962.585      972.983      10.3980
      978.331      972.983      983.534      10.5500
      988.911      983.534      994.152      10.6180
      999.543      994.152      1004.96      10.8060"""
    wv = np.arange ( 400, 2501 )
    bands = np.fromstring ( bands_txt, sep="\n").reshape((62,4))
    proba_min = bands[:, 1]
    proba_max = bands[:, 2]
    band_pass = np.zeros (( 62, 2101 ), dtype = np.bool )
    bw = bands [ :, 3]
    wv = np.arange ( 400, 2501 )
    for i in xrange(62):
        band_pass[i,:] = np.logical_and ( wv >= proba_min[i], \
                          wv <= proba_max[i] )
    return 0.5 * ( proba_min + proba_max ), band_pass

def get_proba ():
    import os
    import gdal
    proba_dir = "/data/netapp_4/ucfajlg/BARRAX/SPARC/" + \
        "SPARC_DB/SPARC2003/PROBA/corregidasAtm/corregidasAtm_120703"
    g = gdal.Open ( os.path.join ( proba_dir, "refl_030712_3598.tif" ) )
    # Typically, one would do this to just get a morsel of the image:
    # proba = g.ReadAsArray()[:,100:300, 100:200]/10000.
    # The first dimension is bands, so can subset there too
    return g



if __name__ == "__main__":
    state_config = OrderedDict ()

    state_config['n'] = CONSTANT
    state_config['cab'] = VARIABLE
    state_config['car'] = FIXED
    state_config['cbrown'] = VARIABLE
    state_config['cw'] = FIXED
    state_config['cm'] = VARIABLE
    state_config['lai'] = VARIABLE
    state_config['ala'] = FIXED
    state_config['bsoil'] = VARIABLE
    state_config['psoil'] = CONSTANT

        
        
        
        
    # Now define the default values
    default_par = OrderedDict ()
    default_par['n'] = 2.
    default_par['cab'] = 40.
    default_par['car'] = 10.
    default_par['cbrown'] = 0.01
    default_par['cw'] = 0.018 # Say?
    default_par['cm'] = 0.0065 # Say?
    default_par['lai'] = 2
    default_par['ala'] = 45.
    default_par['bsoil'] = 1.
    default_par['psoil'] = 0.1

    parameter_min = OrderedDict()
    parameter_max = OrderedDict()
        
    min_vals = [ 0.8, 0.2, 0.0, 0.0, 0.0043, 0.0017,0.01, 40, 0., 0., 0.]
    max_vals = [2.5, 77., 5., 1., 0.0713, 0.0331, 6., 50., 2., 1.]

    for i, param in enumerate ( state_config.keys() ):
        parameter_min[param] = min_vals[i]
        parameter_max[param] = max_vals[i]

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

    # Define the state grid. In time in this case
    state_grid = np.arange ((300*400)).reshape((400,300))
        
    # Define the state
    # L'etat, c'est moi
    state = State ( state_config, state_grid, default_par, \
            parameter_min, parameter_max, verbose=True )
    # Set the transformations
    state.set_transformations ( transformations, inv_transformations )

    ###########################################################
    ### Set priors
    ##########################################################

    mu_prior = OrderedDict ()
    prior_inv_cov = OrderedDict ()
    prior_inv_cov['n'] = np.array([1])
    prior_inv_cov['cab'] = np.array([2.])
    prior_inv_cov['car'] = np.array([1])
    prior_inv_cov['cbrown'] = np.array([1.5])
    prior_inv_cov['cw'] = np.array([1])
    prior_inv_cov['cm'] = np.array([1.5])
    prior_inv_cov['lai'] = np.array([2.])
    prior_inv_cov['ala'] = np.array([1])
    prior_inv_cov['bsoil'] = np.array([3.])
    prior_inv_cov['psoil'] = np.array([1.])
        
    for param in state.parameter_min.iterkeys():
        if transformations.has_key ( param ):
            mu_prior[param] = transformations[param]( \
                    np.array([default_par[param]]) )
        else:
            mu_prior[param] = np.array([default_par[param]])
        prior_inv_cov[param] = 1./prior_inv_cov[param]**2

    prior = Prior ( mu_prior, prior_inv_cov )

    ###########################################################
    ###  MRF prior
    ##########################################################


    spatial= SpatialSmoother ( state_grid, 7, required_params=["lai", "cab", "bsoil", "cm"] )



    ###########################################################
    ###  Create emulators
    ##########################################################
    angles = [ [22, 19, 0] ]

    for i,(s,v,r) in enumerate(angles):     
        fname = "%02d_sza_%02d_vza_000_raa" % (s,v)
        emulators = {}
        if os.path.exists ( fname + ".npz"):
            
            emulators[(v,s)]= MultivariateEmulator ( dump=fname + ".npz" )
            
        else:
            ems, samples, validate = create_emulators ( \
                    state, [""], angles=angles )   
            ems[i].dump_emulator(fname + ".npz" )
            emulators[(v,s)]= MultivariateEmulator ( dump=fname + ".npz" )
            

    the_bands, band_pass = proba_spectral () 
    # The proba data
    g = get_proba ()
    proba_sel_bands = np.array([40, 23, 56, 29, 34,  1,  6, 13])
    data = g.ReadAsArray()[ proba_sel_bands, :, : ]/10000.
    band_pass = band_pass[ proba_sel_bands, :]
    the_bands = the_bands [ proba_sel_bands ]
    mask = np.ones_like ( data[0,:,:], dtype=np.bool )
    #mask[50:75, 100:150 ] = False
    bu = np.ones_like (proba_sel_bands)
    observations = ObservationOperatorImageGP ( state_grid, state, data, mask, emulators[(v,s)], bu, band_pass=band_pass, per_band=True )
    state.add_operator ( "Prior", prior )
    state.add_operator ("MRF", spatial )
    state.add_operator ( "obs", observations )
    print "Starting optimisation"
    print ""
    retval_dict, retval = state.optimize ( x0 = "obs" )
        #{'lai': np.ones_like ( mask )*2., \
                               #'cab': np.ones_like ( mask )*60., \
                               #'cbrown': np.ones_like( mask ) * 0.5, 
                               #'bsoil': np.ones_like ( mask ) * 1.5 })
    