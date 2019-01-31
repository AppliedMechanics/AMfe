
import numpy as np
from h5py import File
from os.path import isfile
from os import remove as fileremove

from amfe.io.tools import *
from amfe.io.mesh.reader import GmshAsciiMeshReader, AmfeMeshObjMeshReader
from amfe.io.mesh.writer import AmfeMeshConverter
from amfe.material import KirchhoffMaterial
from amfe.component import StructuralComponent
from amfe.solver import SolverFactory, AmfeSolution
from amfe.linalg import vector_norm
from amfe.io.postprocessing import *
from amfe.io.postprocessing.tools import write_xdmf_from_hdf5
from amfe.io.postprocessing.reader import AmfeSolutionReader
from amfe.io.postprocessing.writer import Hdf5PostProcessorWriter
from amfe.solver.translators import MechanicalSystem


times = dict([])
input_file = amfe_dir('meshes/gmsh/bar.msh')
output_file = amfe_dir('results/test_refactoring/nonlinear_beam_memoize')

mesh_reader = GmshAsciiMeshReader(input_file)
mesh_converter = AmfeMeshConverter()
mesh_reader.parse(mesh_converter)
mesh = mesh_converter.return_mesh()

my_component = StructuralComponent(mesh)
my_material = KirchhoffMaterial(E=210E9, nu=0.3, rho=1E4, plane_stress=True)
my_component.assign_material(my_material, [7], 'S', '_groups')
dirichlet = my_component._constraints.create_dirichlet_constraint()
# (xy, direction)
my_component.assign_constraint('Leftfixation', dirichlet, [8], '_groups')
neumann = my_component._neumann.create_fixed_direction_neumann((0, -1),
                                                               lambda t: 1E8*np.sin(31*t))

my_component.assign_neumann('Rightneumann', neumann, [9], '_groups')


# ----------------------------------------- NONLINEAR DYNAMIC ANALYSIS ------------------------------------------------
system = MechanicalSystem(my_component)

solfac = SolverFactory()
solfac.set_system(system)
solfac.set_analysis_type('transient')
solfac.set_integrator('genalpha')
solfac.set_large_deflection(True)
solfac.set_nonlinear_solver('newton')
solfac.set_linear_solver('scipy-sparse')
solfac.set_newton_maxiter(30)
solfac.set_newton_atol(1e-6)
solfac.set_newton_rtol(1e-8)
solfac.set_dt_initial(0.001)

residuals = list()


def newton_callback(q, res):
    residuals.append(vector_norm(res))
    u = my_component.unconstrain_vector(q)
    t = 0.0
    du = ddu = np.zeros_like(u)
    my_component._constraints.update_constraints(my_component.X, u, du, ddu, t)


solfac.set_newton_callback(newton_callback)
mysolver = solfac.create_solver()
writer = AmfeSolution()


def write_callback(t, q, dq, ddq):
    u = my_component.unconstrain_vector(q)
    du = my_component.unconstrain_vector(dq)
    ddu = my_component.unconstrain_vector(ddq)
    writer.write_timestep(t, u, du, ddu)


t0 = 0.0
t_end = 1.0
dt = 0.001

no_of_dofs = my_component._constraints.no_of_constrained_dofs
q0 = np.zeros(no_of_dofs)
dq0 = q0.copy()

my_component._constraints.update_constraints(my_component.X, q0, dq0, np.zeros_like(dq0), 0.0)

no_of_dofs = my_component._constraints.no_of_constrained_dofs
q0 = np.zeros(no_of_dofs)
dq0 = q0.copy()


mysolver.solve(write_callback, t0, q0, dq0, t_end)

preader = AmfeSolutionReader(writer, my_component, is_constrained=False)
meshreaderobj = AmfeMeshObjMeshReader(mesh)

hdf5resultsfilename = output_file + '_nonlingenalpha_refactoring' + '.hdf5'
xdmfresultsfilename = output_file + '_nonlingenalpha_refactoring' + '.xdmf'

if isfile(hdf5resultsfilename):
    fileremove(hdf5resultsfilename)

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
        write_xdmf_from_hdf5(xdmffp, hdf5fp, '/mesh/nodes', '/mesh/topology', writer.t, fielddict)

np.save(output_file + '_residuals.npy', np.array(residuals))
