# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

"""
UI module of AMfe.

This is the most simple User Interface for AMfe. It provides some common methods, you'll need to perform a full
Finite Elements simulation. So if you're not familiar with AMfe yet, it is highly recommended to start either with an
example-file in the 'examples'-folder and/or with the methods here. If you want to do something more advanced, you'll
have to dig deeper and take a look at the documentation. No worries, though. Just follow down the line of the
UI-methods.
"""
from amfe.component.structural_component import StructuralComponent
from amfe.material import KirchhoffMaterial, MooneyRivlin
from amfe.forces import *
from amfe.neumann.structural_neumann import FixedDirectionNeumann
from amfe.solver.translators import create_constrained_mechanical_system_from_component
from amfe.solver import SolverFactory, AmfeSolution
from amfe.linalg.linearsolvers import ScipySparseLinearSolver

from amfe.io.mesh import GidJsonMeshReader, AmfeMeshObjMeshReader, GmshAsciiMeshReader
from amfe.io.mesh import AmfeMeshConverter
from amfe.io.postprocessing.reader import AmfeSolutionReader
from amfe.io.postprocessing.writer import Hdf5PostProcessorWriter
from amfe.io.postprocessing.tools import write_xdmf_from_hdf5
from amfe.io.postprocessing import *

import numpy as np
import os
from os.path import splitext
from h5py import File

__all__ = ['import_mesh_from_file',
           'create_structural_component',
           'create_material',
           'assign_material_by_group',
           'assign_material_by_elementids',
           'set_dirichlet_by_group',
           'set_dirichlet_by_nodeids',
           'set_neumann_by_group',
           'set_neumann_by_elementids',
           'solve_linear_static',
           'solve_nonlinear_static',
           'solve_nonlinear_dynamic',
           'write_results_to_paraview',
           ]

formats = {'.json': GidJsonMeshReader,
           '.msh': GmshAsciiMeshReader}


def import_mesh_from_file(filename):
    """
    Loads a Mesh from Filename and converts it to an AMfe Mesh

    Parameters
    ----------
    filename : str
        absolute path to meshfile

    Returns
    -------
    mesh : amfe.mesh.Mesh
        Returns an AMfe Mesh Object
    """
    _, extension = splitext(filename)
    if extension in formats:
        reader = formats[extension](filename)
        converter = AmfeMeshConverter()
        reader.parse(converter)
        return converter.return_mesh()
    else:
        raise ValueError('No reader available for files with fileextension {}'.format(extension), 'Please use a mesh ',
                         'with file-type:', '\n'.join(key for key in formats))


def create_structural_component(mesh):
    return StructuralComponent(mesh)


def create_mechanical_system(structural_component, constant_mass=False,
                             constant_damping=False, constraint_formulation='boolean',
                             **formulation_options):
    system, formulation = create_constrained_mechanical_system_from_component(structural_component, constant_mass,
                                                                              constant_damping, constraint_formulation,
                                                                              **formulation_options)
    return system, formulation


def create_material(material_type='Kirchhoff', **kwargs):
    if kwargs is None:
        print('No material-parameters were given. I am setting them to default.')
    else:
        if material_type is 'Kirchhoff':
            if 'E' in kwargs:
                E = kwargs['E']
            else:
                E = 210E9
            if 'nu' in kwargs:
                nu = kwargs['nu']
            else:
                nu = 0.3
            if 'rho' in kwargs:
                rho = kwargs['rho']
            else:
                rho = 7.85E3
            if 'plane_stress' in kwargs:
                plane_stress = kwargs['plane_stress']
            else:
                plane_stress = True
            if 'thickness' in kwargs:
                thickness = kwargs['thickness']
            else:
                thickness = 1.0

            return KirchhoffMaterial(E, nu, rho, plane_stress, thickness)
        elif material_type is 'MooneyRivlin':
            if 'A10' in kwargs:
                A10 = kwargs['A10']
            else:
                A10 = 0.4E3
            if 'A01' in kwargs:
                A01 = kwargs['A01']
            else:
                A01 = 0.1E3
            if 'kappa' in kwargs:
                kappa = kwargs['kappa']
            else:
                kappa = 1E5
            if 'rho' in kwargs:
                rho = kwargs['rho']
            else:
                rho = 0.94E3
            if 'plane_stress' in kwargs:
                plane_stress = kwargs['plane_stress']
            else:
                plane_stress = True
            if 'thickness' in kwargs:
                thickness = kwargs['thickness']
            else:
                thickness = 1.0

            return MooneyRivlin(A10, A01, kappa, rho, plane_stress, thickness)
        else:
            raise ValueError('Unknown material-type given. Please use one of these supported types: \n ',
                             'Kirchhoff\n',
                             'MooneyRivlin\n'
                             )


def assign_material_by_group(component, material, group_name):
    component.assign_material(material, [group_name], 'S', '_groups')


def assign_material_by_elementids(component, material, elementids):
    component.assign_material(material, elementids, 'S', '_eleids')


def set_dirichlet_by_group(component, group_name, direction=('ux'), constraint_name='Dirichlet0'):
    dirichlet = component.constraints.create_dirichlet_constraint()
    nodeids = component._mesh.get_nodeids_by_groups([group_name])
    supportdofs = component._mapping.get_dofs_by_nodeids(nodeids, direction)
    for dof in supportdofs.reshape(-1):
        component.assign_constraint(constraint_name, dirichlet, np.array([dof], dtype=int), np.array([], dtype=int))


def set_dirichlet_by_nodeids(component, nodeids, direction=('ux'), constraint_name='Dirichlet0'):
    dirichlet = component.constraints.create_dirichlet_constraint()
    supportdofs = component._mapping.get_dofs_by_nodeids(nodeids, direction)
    for dof in supportdofs.reshape(-1):
        component.assign_constraint(constraint_name, dirichlet, np.array([dof], dtype=int), np.array([], dtype=int))


def set_neumann_by_group(component, group_name, direction_vector=np.array([0, 1]), neumann_name='Neumann0',
                         F=constant_force(1.0)):
    neumann = FixedDirectionNeumann(direction_vector, time_func = F)
    component.assign_neumann(neumann_name, neumann, [group_name], '_groups')


def set_neumann_by_elementids(component, elementids, direction_vector=np.array([0, 1]), neumann_name='Neumann0',
                              F=constant_force(1.0)):
    neumann = FixedDirectionNeumann(direction_vector, time_func=F)
    component.assign_neumann(neumann_name, neumann, elementids, '_eleids')


def solve_linear_static(component):
    system, formulation = create_mechanical_system(component, constant_mass=True, constant_damping=True,
                                                   constraint_formulation='boolean')

    solver = ScipySparseLinearSolver()

    no_of_dofs = system.dimension
    q0 = np.zeros(no_of_dofs)
    dq0 = q0

    x = solver.solve(system.K(q0, dq0, 0.0), system.f_ext(q0, dq0, 0.0))

    u, _, _ = formulation.recover(x, q0, q0, 0.0)
    solution_writer = AmfeSolution()
    solution_writer.write_timestep(0.0, u, None, None)

    return solution_writer


def solve_nonlinear_static(component):
    system, formulation = create_mechanical_system(component, constant_mass=True, constant_damping=True,
                                                   constraint_formulation='boolean')

    solfac = SolverFactory()
    solfac.set_system(system)
    solfac.set_analysis_type('static')
    solfac.set_large_deflection(True)
    solfac.set_nonlinear_solver('newton')
    solfac.set_linear_solver('scipy-sparse')
    solfac.set_acceleration_intializer('zero')
    solfac.set_newton_maxiter(30)
    solfac.set_newton_atol(1e-6)
    solfac.set_newton_rtol(1e-8)
    solfac.set_dt_initial(1.0)

    solver = solfac.create_solver()

    solution_writer = AmfeSolution()

    no_of_dofs = system.dimension
    q0 = np.zeros(no_of_dofs)
    dq0 = q0
    t0 = 0.0
    t_end = 1.0

    def write_callback(t, x, dx, ddx):
        u, du, ddu = formulation.recover(x, dx, ddx, t)
        solution_writer.write_timestep(t_end, u, None, None)

    solver.solve(write_callback, t0, q0, dq0, t_end)

    return solution_writer


def solve_nonlinear_dynamic(component, t0, t_end):
    system, formulation = create_mechanical_system(component, constant_mass=True, constant_damping=True,
                                                   constraint_formulation='boolean')

    solfac = SolverFactory()
    solfac.set_system(system)
    solfac.set_analysis_type('transient')
    solfac.set_integrator('genalpha')
    solfac.set_large_deflection(True)
    solfac.set_nonlinear_solver('newton')
    solfac.set_linear_solver('scipy-sparse')
    solfac.set_acceleration_intializer('zero')
    solfac.set_newton_maxiter(30)
    solfac.set_newton_atol(1e-6)
    solfac.set_newton_rtol(1e-8)
    solfac.set_dt_initial(0.001)

    solver = solfac.create_solver()

    solution_writer = AmfeSolution()

    def write_callback(t, x, dx, ddx):
        u, du, ddu = formulation.recover(x, dx, ddx, t)
        solution_writer.write_timestep(t, u, None, None)

    no_of_dofs = system.dimension
    q0 = np.zeros(no_of_dofs)
    dq0 = q0
    solver.solve(write_callback, t0, q0, dq0, t_end)

    return solution_writer


def write_results_to_paraview(solution, component, paraviewfilename):
    """
    Writes results to an xdmf paraview file

    Parameters
    ----------
    solution : amfe.solver.solution.AMfeSolution
        AMfeSolution Object that contains the solution of a time integration for instance
    component : amfe.component.MeshComponent
        MeshComponent Object that contains the information of the simulated MeshComponent the results belong to
    paraviewfilename : str
        Full path to the file were the Paraview results shall be written
    """
    paraviewfilename = splitext(paraviewfilename)[0]

    preader = AmfeSolutionReader(solution, component)
    meshreaderobj = AmfeMeshObjMeshReader(component.mesh)

    hdf5resultsfilename = paraviewfilename + '.hdf5'
    xdmfresultsfilename = paraviewfilename + '.xdmf'

    if os.path.isfile(hdf5resultsfilename):
        os.remove(hdf5resultsfilename)

    pwriter = Hdf5PostProcessorWriter(meshreaderobj, hdf5resultsfilename)
    preader.parse(pwriter)
    pwriter.return_result()

    fielddict = {'displacement': {'mesh_entity_type': MeshEntityType.NODE,
                                  'data_type': PostProcessDataType.VECTOR,
                                  'hdf5path': '/results/displacement'
                                 }
                 }

    with open(xdmfresultsfilename, 'wb') as xdmffp:
        with File(hdf5resultsfilename, mode='r') as hdf5fp:
            write_xdmf_from_hdf5(xdmffp, hdf5fp, '/mesh/nodes', '/mesh/topology', solution.t, fielddict)
