
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
from amfe.io.postprocessing import *
from amfe.io.postprocessing.tools import write_xdmf_from_hdf5
from amfe.io.postprocessing.reader import AmfeSolutionReader
from amfe.io.postprocessing.writer import Hdf5PostProcessorWriter
from amfe.solver.translators import *


times = dict([])
input_file = amfe_dir('meshes/gmsh/bar.msh')
output_file = amfe_dir('results/test_refactoring/nonlinear_beam_new_translators')

mesh_reader = GmshAsciiMeshReader(input_file)
mesh_converter = AmfeMeshConverter()
mesh_reader.parse(mesh_converter)
mesh = mesh_converter.return_mesh()

my_component = StructuralComponent(mesh)
my_material = KirchhoffMaterial(E=210E9, nu=0.3, rho=1E4, plane_stress=True)
my_component.assign_material(my_material, [7], 'S', '_groups')



dirichlet = my_component.constraints.create_dirichlet_constraint()

# Variant A
# (xy, direction)
#nodeids = mesh.get_nodeids_by_groups([8])
#dofs = my_component.mapping.get_dofs_by_nodeids(nodeids, ('ux', 'uy'))
#for dof in dofs.reshape(-1):
    #my_component.assign_constraint('Leftfixation', dirichlet, np.array([dof], dtype=int), np.array([], dtype=int))

#neumann = my_component.neumann.create_fixed_direction_neumann((0, -1),
                                                               #lambda t: 1E8*np.sin(31*t))

#my_component.assign_neumann('Rightneumann', neumann, [9], '_groups')

#q0_raw = np.zeros(my_component.mapping.no_of_dofs)
#dq0_raw = q0_raw.copy()

# Variant B

dirichlet2 = my_component.constraints.create_dirichlet_constraint(lambda t: 3.0*np.sin(31.0*t),
                                                                  lambda t: 3.0*31.0*np.cos(31.0*t),
                                                                  lambda t: -3.0*31.0*31.0*np.sin(31.0*t))

nodeids = mesh.nodes_df.index.values
ydofs = my_component.mapping.get_dofs_by_nodeids(nodeids, ('uy'))
q0_raw = np.zeros(my_component.mapping.no_of_dofs)
dq0_raw = q0_raw.copy()
dq0_raw[ydofs] = 3.0*31.0

# (xy, direction)
nodeids = mesh.get_nodeids_by_groups([8])
supportdofs_x = my_component.mapping.get_dofs_by_nodeids(nodeids, ('ux'))
supportdofs_y = my_component.mapping.get_dofs_by_nodeids(nodeids, ('uy'))
for dof in supportdofs_x.reshape(-1):
    my_component.assign_constraint('Leftfixation', dirichlet, np.array([dof], dtype=int), np.array([], dtype=int))

for dof in supportdofs_y.reshape(-1):
    my_component.assign_constraint('Leftmotion', dirichlet2, np.array([dof], dtype=int), np.array([], dtype=int))
# END Variant B


# ----------------------------------------- NONLINEAR DYNAMIC ANALYSIS ------------------------------------------------

system, formulation = create_constrained_mechanical_system_from_component(my_component, constant_mass=True,
                                                                          constant_damping=True,
                                                                          constraint_formulation='lagrange',
                                                                          scaling=10.0, penalty=3.0)

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

residuals = list()

mysolver = solfac.create_solver()
writer = AmfeSolution()


def write_callback(t, x, dx, ddx):
    u, du, ddu = formulation.recover(x, dx, ddx, t)
    writer.write_timestep(t, u, du, ddu)


t0 = 0.0
t_end = 1.0
dt = 0.001

no_of_dofs = system.dimension
q0 = np.zeros(no_of_dofs)
dq0 = q0.copy()
q0[:len(q0_raw)] = q0_raw
dq0[:len(dq0_raw)] = dq0_raw


mysolver.solve(write_callback, t0, q0, dq0, t_end)

preader = AmfeSolutionReader(writer, my_component)
meshreaderobj = AmfeMeshObjMeshReader(mesh)

hdf5resultsfilename = output_file + '_nonlingenalpha_refactoring_lagrange_scaling_augmented_support_motion' + '.hdf5'
xdmfresultsfilename = output_file + '_nonlingenalpha_refactoring_lagrange_scaling_augmented_support_motion' + '.xdmf'

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
