#!/bin/env python
# -*- coding: utf-8 -*-

"""
Module for advanced nonlinear finite element analysis. This software is developed and maintained at the Chair for Applied Mechanics, Technische Universität München.

Authors:
Johannes Rutzmoser, Fabian Gruber
"""

#import pyximport
#pyximport.install()


#try:
#    from amfe.f90_element import *
#    fortran_use = True
#except:
#    print('''
#Python was not able to load the fast fortran routines.
#run TODO in order to get the full speed!
#''')


from amfe.assembly import *
from amfe.boundary import *
from amfe.element import *
from amfe.mesh import *
from amfe.solver import *
from amfe.tools import *
from amfe.mechanical_system import *

from amfe.model_reduction import *

