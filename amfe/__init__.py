#!/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for advanced nonlinear finite element analysis. This software is
developed and maintained at the Chair for Applied Mechanics,
Technische Universität München.

Authors:
Johannes Rutzmoser, Fabian Gruber
"""
from __future__ import absolute_import

# Finite Element stuff
from .assembly import *
from .boundary import *
from .element import *
from .material import *
from .mechanical_system import *
from .mesh import *
from .solver import *
from .tools import *

# Reduction stuff
from .model_reduction import *
# Commented out as a dill dependency is required only here
# from .num_exp_toolbox import * 


__author__ = 'Johannes Rutzmoser, Fabian Gruber'
