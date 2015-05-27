#!/bin/env python
# -*- coding: utf-8 -*-

"""
Module for finite element analysis

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
# Deprecated as the import gmsh routine is added to the mesh class
# from amfe.import_mesh import *
from amfe.tools import *
from amfe.mechanical_system import *


