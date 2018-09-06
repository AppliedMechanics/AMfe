# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Basic assembly module for the finite element code. Assumes to have all elements in the inertial frame.

Provides an Assembly class which knows the mesh. It can assemble the vector of nonlinear forces, the mass matrix and
the tangential stiffness matrix. Some parts of the code -- mostly the indexing of the sparse matrices -- are
substituted by fortran routines, as they allow for a huge speedup.
"""

from .assembly import Assembly

__all__ = [
    'StructuralAssembly'
]


class StructuralAssembly(Assembly):
    pass
