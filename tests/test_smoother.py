from collections import OrderedDict
import numpy as np
from numpy.testing import assert_equal
from eoldas_ng.operators import TemporalSmoother
#from nose.tools import assert_equal
#from nose.tools import assert_not_equal
#from nose.tools import assert_raises
#from nose.tools import raises

class TestTemporalSmoother ( object ):

    @classmethod
    def setup_class(klass):
        """This method is run once for each class before any tests are run"""
        #state_grid = np.arange ( 1, 365*10, 5 )
        #gamma = 100
        ##self.class = TemporalSmoother ( state_grid, gamma )
        #state_config = OrderedDict ({'magnitude':3})
        
    @classmethod
    def teardown_class(klass):
        """This method is run once for each class _after_ all tests are run"""
        pass

#    def setUp(self):
#        """This method is run once before _each_ test method is executed"""
#
#    def teardown(self):
#        """This method is run once after _each_ test method is executed"""

    def test_gamma(self):
        state_grid = np.arange ( 1, 365*10, 5 )
        gamma = 100
        #self.class = TemporalSmoother ( state_grid, gamma )
        state_config = OrderedDict ({'magnitude':3})

        a = TemporalSmoother ( state_grid, gamma, required_params=['magnitude'] )
        
        assert_equal( a.gamma, gamma)

    def test_Dmatrix_lag(self):
        state_grid = np.arange ( 1, 365*10, 5 )
        gamma = 100
        lag = 1
        N = len(state_grid)
        state_config = OrderedDict ({'magnitude':3})

        a = TemporalSmoother ( state_grid, gamma, lag=lag, required_params=['magnitude'] )
        # Create constraint matrix as main diagonal of ones 
        # and off diagonal shifted by lag of -1s
        I = np.identity( N )
        D1 = np.matrix(I - np.roll( I, lag ))
        D1 =  D1*D1.T
        assert_equal ( a.D1.todense(), D1 )

    def test_cost(self):
        state_grid = np.arange ( 1, 365*10, 5 )
        gamma = 100
        lag = 1
        N = len(state_grid)
        state_config = OrderedDict ({'magnitude':3})
        I = np.identity( N )
        D = np.matrix(I - np.roll( I, lag ))
        x = np.ones_like(state_grid)*3.
        a = TemporalSmoother ( state_grid, gamma, lag=lag, required_params=['magnitude'] )
        x_dict = OrderedDict ({'magnitude':x})
        cost = x.dot ( D*D.T).dot(x)
        cost = 0.5*gamma*cost
        import pdb; pdb.set_trace()
        assert_equal ( np.squeeze(a.der_cost( x_dict, state_config)[0]), cost )

    def test_costa(self):
        state_grid = np.arange ( 1, 365*10, 5 )
        gamma = 100
        lag = 1
        N = len(state_grid)
        state_config = OrderedDict ({'magnitude':3})
        I = np.identity( N )
        D = np.matrix(I - np.roll( I, lag ))
        D1= D*D.T
        x = np.ones_like(state_grid)*3.
        x[23] = 0.
        a = TemporalSmoother ( state_grid, gamma, lag=lag, required_params=['magnitude'] )
        x_dict = OrderedDict ({'magnitude':x*1.})
        cost = np.array( x.dot ( D*D.T).dot(x)).squeeze()
        cost = 0.5*gamma*cost
        
        assert_equal ( np.squeeze(a.der_cost( x_dict, state_config)[0]), cost )


    def test_dcost(self):
        state_grid = np.arange ( 1, 365*10, 5 )
        gamma = 100
        lag = 1
        N = len(state_grid)
        state_config = OrderedDict ({'magnitude':3})
        I = np.identity( N )
        D = np.matrix(I - np.roll( I, lag ))
        x = np.ones_like(state_grid)*3.
        a = TemporalSmoother ( state_grid, gamma, lag=lag, required_params=['magnitude'] )
        x_dict = OrderedDict ({'magnitude':x})
        cost = gamma*(D.dot(x)).T.dot(D.dot(x))
        assert_equal ( np.squeeze(a.der_cost( x_dict, state_config)[1]), np.zeros(N) )
        
    def test_dcosta(self):
        state_grid = np.arange ( 1, 365*10, 5 )
        gamma = 100
        lag = 1
        N = len(state_grid)
        state_config = OrderedDict ({'magnitude':3})
        I = np.identity( N )
        D = np.matrix(I - np.roll( I, lag ))
        x = np.ones_like(state_grid)*3.
        x[23] = 100
        a = TemporalSmoother ( state_grid, gamma, lag=lag, required_params=['magnitude'] )
        x_dict = OrderedDict ({'magnitude':x*1.})
        dcost = gamma*np.array((D*D.T)).dot(x)
        assert_equal ( np.squeeze(a.der_cost( x_dict, state_config)[1]), dcost )
