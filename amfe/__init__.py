#!/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for advanced nonlinear finite element analysis. This software is
developed and maintained at the Chair for Applied Mechanics,
Technische Universität München.

Authors:
Johannes Rutzmoser, Fabian Gruber
"""

# Finite Element stuff
from amfe.assembly import Assembly
from amfe.boundary import DirichletBoundary, NeumannBoundary
from amfe.element import Element, Tri3, Tri6, Quad4, Quad8, Tet4, Tet10, \
    Bar2Dlumped, BoundaryElement, Tri3Boundary, Tri6Boundary, \
    LineLinearBoundary, LineQuadraticBoundary
from amfe.material import HyperelasticMaterial, KirchhoffMaterial, NeoHookean, \
    MooneyRivlin
from amfe.mechanical_system import MechanicalSystem, ReducedSystem, QMSystem, \
    ConstrainedMechanicalSystem
from amfe.mesh import Mesh, MeshGenerator
from amfe.solver import NewmarkIntegrator, solve_linear_displacement, \
    solve_nonlinear_displacement, give_mass_and_stiffness, HHTConstrained
from amfe.tools import *

# Reduction stuff
from amfe.model_reduction import *
