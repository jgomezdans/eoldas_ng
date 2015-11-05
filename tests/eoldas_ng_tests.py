from nose.tools import *
import eoldas_ng
import numpy as np
import scipy.sparse as sp
from collections import OrderedDict

#def setup():
    #print "SETUP!"

#def teardown():
    #print "TEAR DOWN!"

#def test_basic():
    #print "I RAN!"

class TestPrior: 
    def setUp ( self ):
        state_config = { 'magnitude': eoldas_ng.VARIABLE }
        
        default_values = { 'magnitude': 0.5 }
        parameter_min = OrderedDict ()
        parameter_max = OrderedDict ()
        parameter_min [ 'magnitude' ] = 0.
        parameter_max [ 'magnitude' ] = 1.
        # A daily time series with 1 day sampling
        state_grid = np.arange ( 1, 366 ) 
        self.the_state = eoldas_ng.State ( state_config, state_grid, 
                                          default_values, 
                           parameter_min, parameter_max )
        mu_prior = { 'magnitude': np.array([0.5]) }
        inv_cov_prior = { 'magnitude': np.array([1./(2*2)]) }
        the_prior = eoldas_ng.Prior ( mu_prior, inv_cov_prior ) 
        
        self.the_state.add_operator ("Prior", the_prior )
        
        
    def test_prior_cost_using_mean ( self ):
        x_test = np.ones(365)*0.5
        cost, dcost = self.the_state.cost ( x_test )
        assert cost == 0.

    def test_prior_cost_not_using_mean ( self ):
        x_test = np.zeros(365)
        cost, dcost = self.the_state.cost ( x_test )
        assert cost == 0.5*0.25*365/4.

    def test_prior_dcost_not_using_mean ( self ):
        x_test = np.zeros(365)
        cost, dcost = self.the_state.cost ( x_test )
        assert np.alltrue( dcost == np.ones_like (x_test) *(-0.5/4.))


    def test_prior_gradient ( self ):
        x_test = np.ones(365)*0.5
        cost, dcost = self.the_state.cost ( x_test )
        assert np.alltrue ( dcost == 0. )

        
    def test_instantiate_with_dict ( self ):
        mean = np.ones(10)
        sd = sp.dia_matrix ( (np.ones(10), 0), shape=(10,10))
        prior = eoldas_ng.Prior ( mean, sd )
        assert ( np.allclose(prior.mu, mean ) )
    #def test_calculation ( self ):
        