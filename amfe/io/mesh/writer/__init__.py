#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Mesh Output module.

Module handling Mesh Output Operations
"""


# --- MESH CONVERTER ---
from .amfe_mesh_converter import *
from .hdf5_mesh_converter import *
from .vtk_mesh_converter import *
