#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Mesh I/O module.

Module handling Mesh I/O Operations
"""


# --- MESH READER --
from .amfe_mesh_obj_mesh_reader import *
from .gid_ascii_mesh_reader import *
from .gid_json_mesh_reader import *
from .gmsh_ascii_mesh_reader import *
from .hdf5_mesh_reader import *
from .gmsh_ascii_v4_mesh_reader import *
