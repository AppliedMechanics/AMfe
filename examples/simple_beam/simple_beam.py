
from amfe.io import AmfeMeshConverter, GidJsonMeshReader
from amfe.tools import amfe_dir
from amfe.material import KirchhoffMaterial
from amfe.component import StructuralComponent
from amfe.neumann import FixedDirectionNeumann
import logging
import numpy as np

from amfe.mesh import Mesh

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


input_file = amfe_dir('meshes/amfe_json/simple_beam/simple_beam.json')
my_mesh = GidJsonMeshReader(input_file, AmfeMeshConverter()).parse()

my_material = KirchhoffMaterial(E_alu, nu_alu, rho_alu, thickness=10)

my_component = StructuralComponent(my_mesh)

my_component.assign_material(my_material, 'Quad8', 'S', 'shape')

my_neumann = FixedDirectionNeumann(np.array([0, 1]), time_func = lambda t: 2)
my_component.assign_neumann('Neumann0', my_neumann, ['right_boundary'], '_groups')

my_constraint = my_component.constraints.create_dirichlet_constraint()
fixed_nodeids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)
my_component.assign_constraint('Dirichlet0', my_constraint, fixed_nodeids, '_nodeids', 'elim')


print('END')