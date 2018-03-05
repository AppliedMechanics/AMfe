#!/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Module for advanced nonlinear finite element analysis. This software is developed and maintained at the Chair for
Applied Mechanics, Technical University of Munich.

Authors:
Christian Meyer, Christopher Lerch, Johannes Rutzmoser, Fabian Gruber
"""
from __future__ import absolute_import

# Finite Element stuff
from .assembly import *
from .boundary import *
from .element import *
from .material import *
from .mechanical_system import *
from .mesh import *
from .observers import *
from .solver import *
from .tools import *

# Linalg submodule
from .linalg import *

# Reduction stuff
from .reduced_basis import *
from .hyper_red import *

# Structural dynamics
from .structural_dynamics import *

# Commented out as a dill dependency is required only here
from .num_exp_toolbox import *

__author__ = 'Christian Meyer, Christopher Lerch, Johannes Rutzmoser, Fabian Gruber'
