
from h5py import File
from os.path import isfile
from os import remove as fileremove

from amfe.io import amfe_dir
from amfe.io.mesh import AmfeMeshConverter, GmshAsciiMeshReader, AmfeMeshObjMeshReader, VtkMeshConverter
from amfe.io.postprocessing.reader import AmfeSolutionReader
from amfe.io.postprocessing.writer import Hdf5PostProcessorWriter
from amfe.io.postprocessing.constants import *
from amfe.io.postprocessing.tools import write_xdmf_from_hdf5
from amfe.solver import SolverFactory, AmfeSolution
from amfe.material import KirchhoffMaterial
from amfe.component import StructuralComponent
from amfe.component.tree_manager import TreeBuilder
from amfe.component.component_composite import MeshComponentComposite
from amfe.neumann import FixedDirectionNeumann
from amfe.solver.domain_decomposition_solver import FETISolver
import logging
import numpy as np
from amfe.solver.translators import MulticomponentMechanicalSystem
# Units:
#   Length: mm
#   Mass:   g
#   Time:   s
#
# Derived Units:
#   Force:  g mm s-2 = µN
#   Stiffness: g s-2 mm-1 = Pa
#   velocity: mm/s
#   acceleration: mm/s^2
#   density: g/mm3

E_alu = 70e6
nu_alu = 0.34
rho_alu = 2.7e-3

logging.basicConfig(level=logging.DEBUG)


input_file = amfe_dir('meshes/gmsh/simple_beam_metis_10/simple_beam_metis_10.msh')
mesh_reader = GmshAsciiMeshReader(input_file)
mesh_converter = AmfeMeshConverter()
mesh_reader.parse(mesh_converter)
my_mesh = mesh_converter.return_mesh()

my_material = KirchhoffMaterial(E_alu, nu_alu, rho_alu, thickness=10)

my_component = StructuralComponent(my_mesh)

material_tag = ['material']
my_component.assign_material(my_material, material_tag, 'S', '_groups')

glo_dofs_x = my_component.mapping.get_dofs_by_nodeids( my_component.mesh.get_nodeids_by_groups(['dirichlet']), ('ux') )
glo_dofs_y = my_component.mapping.get_dofs_by_nodeids( my_component.mesh.get_nodeid_by_coordinates(0.0, 0.0, 0.0), ('uy') )
my_composite = MeshComponentComposite(my_component)
# Decomposition of component
tree_builder = TreeBuilder()
tree_builder.add([0], [my_composite])
leaf_id = tree_builder.leaf_paths.max_leaf_id
tree_builder.separate_partitioned_component_by_leafid( leaf_id )

structural_composite = tree_builder.root_composite.components[0]

# Neumann conditions
my_neumann = FixedDirectionNeumann(np.array([0, 1]), time_func = lambda t: -5.0e4)
structural_composite.assign_neumann('Neumann0', my_neumann, ['neumann'], '_groups')

# Dirichlet conditions
dirichlet = structural_composite.components[1]._constraints.create_dirichlet_constraint()
for dof in glo_dofs_x.reshape(-1):
    structural_composite.assign_constraint('Dirichlet0', dirichlet, np.array([dof], dtype=int), [])
for dof in glo_dofs_y.reshape(-1):
    structural_composite.assign_constraint('Dirichlet1', dirichlet, np.array([dof], dtype=int), [])

# FETI-solver
substructured_system = MulticomponentMechanicalSystem(structural_composite, 'boolean')

fetisolver = FETISolver()
q_dict = fetisolver.solve(substructured_system.mechanical_systems, substructured_system.connections)
u_dict, du_dict, ddu_dict = substructured_system.recover(q_dict)

# Exporter
for i_comp, comp in structural_composite.components.items():
    writer = AmfeSolution()
    writer.write_timestep(0, u_dict[i_comp], None, None)

    mesh = comp._mesh
    preader = AmfeSolutionReader(writer, comp, is_constrained=False)
    amfeMeshReader = AmfeMeshObjMeshReader(mesh)
    path = amfe_dir('meshes/gmsh/simple_beam_metis_10/results/partition_mesh_' + str(i_comp) + '.vtk')
    VTKwriter = VtkMeshConverter(path)

    amfeMeshReader.parse(VTKwriter)
    VTKwriter.return_mesh()

    hdf5resultsfilename = amfe_dir('meshes/gmsh/simple_beam_metis_10/results/partition_mesh_' + str(i_comp) + '.hdf5')
    xdmfresultsfilename = amfe_dir('meshes/gmsh/simple_beam_metis_10/results/partition_mesh_' + str(i_comp) + '.xdmf')
    if isfile(hdf5resultsfilename):
        fileremove(hdf5resultsfilename)
    amfeMeshReader = AmfeMeshObjMeshReader(mesh)
    pwriter = Hdf5PostProcessorWriter(amfeMeshReader, hdf5resultsfilename)
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


print('END')