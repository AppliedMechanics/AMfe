#!/bin/env python
# -*- coding: utf-8 -*-

"""
Module for advaned nonlinear finite element analysis developed at the Chair for Applied Mechanics, Technische Universität München.

Authors:
Johannes Rutzmoser
Fabian Gruber

"""

import pyximport
pyximport.install()



from amfe.assembly import *
from amfe.boundary import *
from amfe.element import *
from amfe.mesh import *
from amfe.solver import *
from amfe.tools import *
from amfe.mechanical_system import *

from amfe.model_reduction import *

