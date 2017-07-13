#!/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Module for advanced nonlinear finite element analysis. This software is
developed and maintained at the Chair for Applied Mechanics,
Technical University of Munich.

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
from .num_exp_toolbox import *

# from .hyper_red import *

__author__ = 'Johannes Rutzmoser, Fabian Gruber'
