from collections import OrderedDict


import numpy as np
import scipy.sparse as sp

from eoldas_ng import *

from eoldas_ng import Prior, SpatialSmoother
from eoldas_ng import ObservationOperatorImageGP, StandardStatePROSAIL
import gp_emulator
from eoldas_observation_helpers import *
        
def set_prior (state, prev_date=None ):
    mu_prior = OrderedDict (  )
    prior_inv_cov = OrderedDict ()
    prior_inv_cov['n'] = np.array([1])
    prior_inv_cov['cab'] = np.array([1])
    prior_inv_cov['car'] = np.array([1])
    prior_inv_cov['cbrown'] = np.array([1])
    prior_inv_cov['cw'] = np.array([0.7])
    prior_inv_cov['cm'] = np.array([0.7])
    prior_inv_cov['lai'] = np.array([1])
    prior_inv_cov['ala'] = np.array([1])
    prior_inv_cov['bsoil'] = np.array([3])
    prior_inv_cov['psoil'] = np.array([3])
        
    for param in state.parameter_min.iterkeys():
        if state.transformations.has_key ( param ):
            mu_prior[param] = state.transformations[param]( \
                    np.array([default_par[param]]) )
        else:
            mu_prior[param] = np.array([default_par[param]])
        prior_inv_cov[param] = 1./prior_inv_cov[param]**2

    n_elems = state_grid.size 
    n_pars = 0
    for k,v in state.state_config.iteritems():
        if v == 3: # VARIABLE for the time being...
            n_pars += 1
            
    prior_vect = np.zeros ( n_elems*n_pars  )# 
    inv_sig_prior = np.zeros ( n_elems*n_pars  )# 6 variable params + 2 const
    # Now, populate said vector in the right order
    # looping over state_config *should* preserve the order
    if prev_date is not None:
        f = cPickle.load ( open ( prev_date, 'r'))
        x = f['transformed_map'] # needs to be vectorised, it's a dict
        prior_vect = state.pack_from_dict ( x, do_transform=False )
        prior_inv_covariance = f['hessian']
        
    else:    
        i = 0
        for param, typo in state_config.iteritems():
            if typo == CONSTANT: # Constant value for all times
                if prev_date is None:
                    prior_vect[i] = mu_prior[param]
                else:
                    prior_vect[i] = previous_posterior['transformed_map'][param]
                inv_sig_prior[i] = prior_inv_cov[param]
                i = i+1        
            elif typo == VARIABLE:
                # For this particular date, the relevant parameter is at location iloc
                if prev_date is None:
                    prior_vect[i:(i + n_elems)] =  \
                        np.ones(n_elems)*mu_prior[param]
                else:
                    prior_vect[i:(i + n_elems)] = \
                        previous_posterior['transformed_map'][param].flatten()
                inv_sig_prior[i:(i + n_elems)] = \
                        np.ones ( n_elems)*prior_inv_cov[param]
                i += n_elems
        prior_inv_covariance = sp.dia_matrix( ([inv_sig_prior], [0]),\
            shape=[n_elems*n_pars, n_elems*n_pars])


    prior = Prior ( prior_vect, prior_inv_covariance )
    return prior


    
class SingleImageProcessor ( object ):
    """This class sets up and solves a single date/single image using 
    eoldas_ng"""

    def __init__ ( self, state_config, state_grid, image, band_unc, mask, \
        bw, band_pass, emulator, prior, regularisation, process_name, \
        factor, optimisation_options=None, verbose=False ):
        """The class creator. Takes state configuration and grids (spatial),
        band uncertainties, image and associated mask, relevant emulator, a
        prior object and a regularisation (spatial) model. We also have a 
        process name to store the results more or less neatly.

        TODO We ought to have private methods to set up the different 
        objects, as that allows the user to select a different type of
        "pre-packaged" state (e.g. not the prosail one).
        
        Parameters
        -----------
        state_config: dict
            A state configuration object
        state_grid: array
            A state grid (2D)
        image: array
            A multispectral image (3D)
        band_unc: array
            Per band uncertainty. Note that I need to check whether you
            can have a per pixel uncertainty, or whether that requires
            some extra recoding. CHECK
        mask: array
            A 2D array, indicating pixels which are OK and pixels which 
            aren't
        bw: array?
            A BW array. This is needed for somem reason, but i think it's
            superfluous CHECK
        band_pass: list
            A list of band_pass objects. We use this list (and the ``bw``
            parameter above) to calculate per band emulators and the 
            initial inverse emulators.
        emulator: list
            The emulator for this particular image acquisition geometry.
            In principle this is a MultivariateEmulator in a single element 
            list, but could be per band emulators
        prior: Prior
            An ``eoldas_ng`` prior object
        regularisation: SpatialSmoother
            An ``eoldas_ng`` SpatialSmoother object.
        process_name: str 
            A string with the process name. This is used to store the 
            results of inverting the current image
        factor: int
            The spatial factor: how many times does this observation
            fit in with the state spatial resolution?
        optimisation_options: dict
            A dictionary of optimisation options
        verbose: bool
            A verbose flag            
        """
        
        # Get the state [CENSORED]
        self.the_state = StandardStatePROSAIL ( state_config, state_grid, \
                 optimisation_options=optimisation_options, \
                 output_name=process_name, verbose=verbose )
        self._build_observation_constraint ( state_grid, image, mask, 
                                            emulator, band_unc, factor,
                                            band_pass, bw )
        # Note that this isn't very satisfactory, but we can always change 
        # that: the prior just needs a mean and a sparse covariance, and 
        # the regularisation needs the gamma(s) and parameters to apply it
        # to
        self.the_prior = prior
        self.the_model = regularisation
        print "WARNING! No prior involved here!"
        print "PRIOR needs defining!"
        #self.the_state.add_operator ( "Prior", self.the_prior )
        self.the_state.add_operator ( "Regularisation", self.the_model )
        self.the_state.add_operator ( "Observations", self.the_observations )
        
    def _build_observation_constraint ( self, state_grid, image, image_mask, \
        emulator, band_uncertainty, factor, band_pass, bw, per_band=True):
        """This is just a method to build up the observation operator."""
        
        self.the_observations = ObservationOperatorImageGP ( state_grid, \
                self.the_state, image, image_mask, emulator, \
                bu=band_uncertainty, factor=factor, \
                band_pass=band_pass, bw=bw, per_band=per_band )
    def first_guess ( self ):
        
        return self.the_observations.first_guess ( \
            self.the_state.state_config )

    def solve ( self, x0=None ):
        return self.the_state.optimize ( x0=x0, do_unc=True )
    
            
"""Some requirements for processing:
* We need a a state config, both for each image and temporally, as we might 
need to have parametes with no temporal correlation (atmosphere). So we need
a ``time_state_config`` and ``space_state_config``. The complication here is
that the Hessian needs to be modified to revert to the prior for the 
parameters that do not vary in time, which is a pain in the arse.

* There can be more than ons observation per time step. This helps with e.g.
atmospheric characterisation

* We might need to rearrange the individual posterior state dumps in some
easy to use/visualise GDAL geo files.

* I can't quite remember the update equation for the Hessian ;-)
Here it is: If we have the posterior Hessian, $\mathbf{A}^{-1}$, and the inflation 
matrix, $\mathbf{B}$, then the resulting "inflated" matrix is
$$
\begin{align}
\mathbf{C}_{prior}^{-1}&=(\mathbf{A} + \mathbf{B})^{-1} \\
    =& \mathbf{A}^{-1}-\mathbf{A}^{-1}\mathbf{B}\left[\mathbf{I}+\mathbf{A}^{-1}\mathbf{B}\right]^{-1}\mathbf{A}^{-1}\\
\end{align}
$$

or, in python parlance...
a_i - a_i.dot(b).dot ( np.linalg.inv(np.eye(5) + a_i.dot(b))).dot(a_i)

How does this work with a sparse matrix is still open to question...

"""

class SpaceTimeProcessor ( object ):

    def __init__ ( self, start_time, end_time, time_resolution, space_grid, \
            observations, space_gamma, time_gamma ):
        
        self.time_grid = time_grid
        self.space_grid = space_grid
        self.observations = observations
        self.space_gamma = space_gamma
        self.time_gamma = time_gamma
        
    def initialise_process ( self ):
        """This method initialises the entire process """
        
        self.prior = set_prior (state, prev_date=None )
        self.smoother = SpatialSmoother ( self.space_grid, self.space_gamma, \
            required_params=["lai", "cab", "cw", "cm", "psoil", "bsoil"] )

    def inflate_uncertainty ( hessian ):
        N = a_i.shape[0]
        # b here is inflation matrix, should just be diagonal, right?
        return hessian - hessian.dot(b).dot ( np.linalg.inv(sp.eye(N) + a_i.dot(b))).dot(a_i)
    
if __name__ == "__main__":
#    obs1 = ObservationStorage ( "/storage/ucfajlg/MidiPyrenees")
    state_grid = np.zeros ( ( 584, 349 ) )
    
    state_config = OrderedDict ()
    state_config['n'] = FIXED
    state_config['cab'] = VARIABLE
    state_config['car'] = FIXED
    state_config['cbrown'] = FIXED
    state_config['cw'] = VARIABLE
    state_config['cm'] = VARIABLE
    state_config['lai'] = VARIABLE
    state_config['ala'] = FIXED
    state_config['bsoil'] = VARIABLE
    state_config['psoil'] = VARIABLE

    spatial_factor = None#[1,1]
    space_gamma = 0.5
    the_smoother = SpatialSmoother ( state_grid, space_gamma, \
            required_params=["lai", "cab", "cw", "cm", "psoil", "bsoil"] )
#    the_prior = set_prior ()
    
    resample_opts = {'box': [546334.113775153,  6274489.49408634,  \
        558032.21126551,  6267491.58431377]}
    bu = [ 0.02, 0.02, 0.02, 0.02]
    spot_observations = SPOTObservations( "/storage/ucfajlg/MidiPyrenees/SPOT", resample_opts )
    for s in spot_observations.loop_observations ( "2013-01-01", "2013-12-31" ):
        if s.have_obs: 
            print s.date
            # First element of iterator is True, then we have an observation
            # What is needed at this stage is
            # 1. The prior object (see ``set_prior`` function)
            # 2. The spatial smoother
            # 3. A state configuration dictionary
            # 4. A state grid
            # 5. The spatial factor (ratio of observation spatial resolution
            #    to state spatial resolution. Must be integer)
            # 6. The actual data array 
            # 7. The relevant mask
            # 8. The observational uncertainty
            # 9. Spectral parameters
            # 10. Process name from the actual image filename
            this_image = SingleImageProcessor ( state_config, state_grid,
                                               s.image/1000., bu, s.mask, s.spectrum.bw,
                                               s.spectrum.band_pass, s.emulator, the_smoother, 
                                               the_smoother, s.fname.replace("_crop.tif", ""),
                                               spatial_factor )
            x0 = this_image.first_guess ()
            
            break
                ###def __init__ ( self, state_config, state_grid, image, band_unc, mask, \
        ###bw, band_pass, emulator, prior, regularisation, process_name, \
        ###factor, optimisation_options=None, verbose=False ):

        # Just some random text added at the end to check pre-commit hooks
        
        
