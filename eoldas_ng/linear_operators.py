#!/usr/bin/env python
"""
A set of classes that implement linear operators are weak constraints
for eoldas_ng. The aim of this class is to implement the following type
of models:
    
$$
p_1 = m*p_2 + c + \eta_{model},
$$
so that we cast $p_1$ as a linear function of $p_2$, with some additional
uncertainty. The aim of the implementation is to have a prior for $m$ and $c$
and update them as the observations see fit.
"""

import numpy as np
import scipy.sparse as sp

FIXED = 1
CONSTANT = 2
VARIABLE = 3

class LinearOperator ( object ):
    """
    A set of classes that implement linear operators are weak constraints
    for eoldas_ng. The aim of this class is to implement the following type
    of models:

    $$
    p_1 = m*p_2 + c + \eta_{model},
    $$
    so that we cast $p_1$ as a linear function of $p_2$, with some additional
    uncertainty. The aim of the implementation is to have a prior for $m$ and $c$
    and update them as the observations see fit.
    """
    def __init__ ( self, p1, p2, m, c, model_unc ):
        """We take two parameters, ``p1`` and ``p2``, where ``p1`` is a linear
        mapping of ``p2``, governed by ``m`` and ``c``. This model has an uncertainty
        given by ``model_unc``.""" 
        
        self.p1 = p1
        self.p2 = p2
        self.m = m
        self.c = c
        self.model_unc = model_unc
        
    def der_cost ( self, x_dict, state_config ):
        """The cost function and associated gradient"""
        p1 = x_dict[self.p1]
        p2 = x_dict[self.p2]
        m = x_dict[self.m]
        c = x_dict[self.c]
        # Figure out problem size
        n = 0
        for param, typo in state_config.iteritems():
            if typo == CONSTANT:
                n += 1
            elif typo == VARIABLE:
                n_elems = len ( x_dict[param] )
                n += n_elems
        
        x_params = np.empty ( ( len( x_dict.keys()), n_elems ) )
        err = p1 - m*p2 - c 
        cost = ( -0.5*  err**2/self.model_unc**2 ).sum()
        der_cost = np.zeros ( n )
        for param, typo in state_config.iteritems():
            
            if typo == FIXED: # Default value for all times
                # Doesn't do anything so we just skip
                pass
                
            elif typo == CONSTANT: # Constant value for all times
                if param == self.p1:
                    der_cost[i] = -err/model_unc**2
                elif param == self.p2:
                    der_cost[i] = m*err/model_unc**2
                elif param == self.m:
                    der_cost[i] = p2*err/model_unc**2
                elif param == self.c:
                    der_cost[i] = err/model_unc**2
                i += 1                
            elif typo == VARIABLE:
                ## TODO Really unsure about this!
                if param == self.p1:
                    der_cost[i:(i+n_elems)] = -err/model_unc**2
                elif param == self.p2:
                    der_cost[i:(i+n_elems)] = m*err/model_unc**2
                elif param == self.m:
                    der_cost[i:(i+n_elems)] = p2*err/model_unc**2
                elif param == self.c:
                    der_cost[i:(i+n_elems)] = err/model_unc**2

                i += n_elems

        return cost, der_cost
    
    def der_der_cost ( x, state_config ):
        """ TODO Need to sort out the component ordering, but I need a better
        understanding of the CONSTANT vs VARIABLE things before this is 
        good enough."""
        #hessian = sp.lil_matrix ()