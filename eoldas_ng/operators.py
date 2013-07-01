#!/usr/bin/env python
"""
EOLDAS ng
==========

A reorganisation of the EOLDAS codebase

"""

import numpy as np
import scipy.optimize

import scipy.ndimage.interpolation

class CostOperator ( object ):
    """A barebones cost function class

    This class is an abstraction of the typical cost function that one might
    use for building up a complex and complete set of operators to use in
    a variational DA scheme.
    """
    def __init__ ( self ):

    def calc_cost ( self, x ):

    def calc_cost_hess ( self, x ):
