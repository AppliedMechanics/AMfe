"""Test Routine for structural component"""


from unittest import TestCase
import numpy as np
from scipy.linalg import norm
from numpy.testing import assert_allclose
from copy import deepcopy

from amfe.io.tools import amfe_dir
from amfe.io.mesh.reader import GidJsonMeshReader
from amfe.io.mesh.writer import AmfeMeshConverter
from amfe.material import KirchhoffMaterial
from amfe.component.structural_component import StructuralComponent


class StructuralComponentTest(TestCase):
    def setUp(self):
        mesh_input = amfe_dir('tests/meshes/gid_json_4_tets.json')
        mesh_reader = GidJsonMeshReader(mesh_input)
        converter = AmfeMeshConverter()
        mesh_reader.parse(converter)
        self.mesh = converter.return_mesh()
        self.amp = 1.0
        my_comp = StructuralComponent(self.mesh)
        my_material = KirchhoffMaterial()
        my_comp.assign_material(my_material, ['left', 'right'], 'S')
        neumann_bc = my_comp._neumann.create_fixed_direction_neumann((1, 0), lambda t: self.amp*t)
        my_comp.assign_neumann('Right force', neumann_bc, ['right_boundary'])
        self.my_comp = my_comp

    def test_f_ext(self):
        q = dq = ddq = np.zeros(self.my_comp._constraints.no_of_constrained_dofs)
        f_ext = self.my_comp.f_ext(q, dq, ddq, 1.0)

        summed_force_actual = np.sum(f_ext)
        length_right = norm(self.mesh.nodes_df.loc[15] - self.mesh.nodes_df.loc[13])
        summed_force_desired = self.amp * length_right
        assert_allclose(summed_force_actual, summed_force_desired)
        # test global locations of f_ext
        locations_not_zero_desired = self.my_comp._mapping.nodal2global.loc[[13, 14, 15], 'ux']
        locations_not_zero_actual = np.nonzero(f_ext)[0]
        self.assertTrue(np.all(np.isin(locations_not_zero_desired, locations_not_zero_actual)))
        self.assertTrue(np.all(np.isin(locations_not_zero_actual, locations_not_zero_desired)))
        self.assertEqual(len(locations_not_zero_desired), len(locations_not_zero_actual))

        # check if time parameter is propagated:
        f_ext = deepcopy(self.my_comp.f_ext(q, dq, ddq, t=1.0))
        f_ext_2 = self.my_comp.f_ext(q, dq, ddq, t=2.0)

        assert_allclose(2*f_ext, f_ext_2)
