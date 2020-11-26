"""Test Routine for structural component"""


from unittest import TestCase
from os.path import join, dirname, abspath
import numpy as np
from scipy.linalg import norm
from numpy.testing import assert_allclose, assert_array_almost_equal
from copy import deepcopy

from amfe.io.tools import amfe_dir
from amfe.io.mesh.reader import GidJsonMeshReader, GmshAsciiMeshReader
from amfe.io.mesh.writer import AmfeMeshConverter
from amfe.material import KirchhoffMaterial
from amfe.component.structural_component import StructuralComponent
from amfe.mesh import Mesh

from amfe.solver import AmfeSolution
import amfe.ui as ui


class StructuralComponentTest(TestCase):
    def setUp(self):
        here = dirname(abspath(__file__))
        mesh_input = join(here, 'meshes', 'gid_json_4_tets.json')
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
        q = dq = ddq = np.zeros(self.my_comp.constraints.no_of_dofs_unconstrained)
        f_ext = self.my_comp.f_ext(q, dq, 1.0)

        summed_force_actual = np.sum(f_ext)
        length_right = norm(self.mesh.nodes_df.loc[15] - self.mesh.nodes_df.loc[13])
        summed_force_desired = self.amp * length_right
        assert_allclose(summed_force_actual, summed_force_desired)
        # test global locations of f_ext
        locations_not_zero_desired = self.my_comp.mapping.nodal2global.loc[[13, 14, 15], 'ux']
        locations_not_zero_actual = np.nonzero(f_ext)[0]
        self.assertTrue(np.all(np.isin(locations_not_zero_desired, locations_not_zero_actual)))
        self.assertTrue(np.all(np.isin(locations_not_zero_actual, locations_not_zero_desired)))
        self.assertEqual(len(locations_not_zero_desired), len(locations_not_zero_actual))

        # check if time parameter is propagated:
        f_ext = deepcopy(self.my_comp.f_ext(q, dq, t=1.0))
        f_ext_2 = self.my_comp.f_ext(q, dq, t=2.0)

        assert_allclose(2*f_ext, f_ext_2)

    def test_fields(self):
        fields_actual = self.my_comp.fields
        fields_desired = ['ux', 'uy']
        self.assertListEqual(fields_actual, fields_desired)

    def test_strains_and_stresses(self):
        # Rigid body-movement not causing strains and stresses
        q = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        dq = np.zeros(self.my_comp.constraints.no_of_dofs_unconstrained)
        strains_actual, stresses_actual = self.my_comp.strains_and_stresses(q, dq, 0.0)

        strains_desired = np.zeros((15, 6))
        stresses_desired = np.zeros((15, 6))

        assert_array_almost_equal(strains_actual, strains_desired, 8)
        assert_array_almost_equal(stresses_actual, stresses_desired, 4)

        # 1D-deformation resulting in 1D-strains and -stresses
        # 2 Quad4-Elements
        here = dirname(abspath(__file__))
        mesh_input = join(here, 'meshes', 'gmsh_2_quads.msh')
        mesh_reader = GmshAsciiMeshReader(mesh_input)
        converter = AmfeMeshConverter()
        mesh_reader.parse(converter)
        mesh = converter.return_mesh()
        amp = 1.0
        my_comp = StructuralComponent(mesh)

        my_material = KirchhoffMaterial(E=30e6, nu=0.0)
        my_comp.assign_material(my_material, ['volume'], 'S')

        q = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.2, 0.0, 0.2, 0.0])
        dq = np.zeros(my_comp.constraints.no_of_dofs_unconstrained)
        strains_actual, stresses_actual = my_comp.strains_and_stresses(q, dq, 0.0)

        strains_desired = np.array([[1.05e-1, 0, 0, 0, 0, 0],
                                    [1.05e-1, 0, 0, 0, 0, 0],
                                    [1.05e-1, 0, 0, 0, 0, 0],
                                    [1.05e-1, 0, 0, 0, 0, 0],
                                    [1.05e-1, 0, 0, 0, 0, 0],
                                    [1.05e-1, 0, 0, 0, 0, 0]])

        stresses_desired = np.array([[3.15e6, 0, 0, 0, 0, 0],
                                    [3.15e6, 0, 0, 0, 0, 0],
                                    [3.15e6, 0, 0, 0, 0, 0],
                                    [3.15e6, 0, 0, 0, 0, 0],
                                    [3.15e6, 0, 0, 0, 0, 0],
                                    [3.15e6, 0, 0, 0, 0, 0]])

        assert_array_almost_equal(strains_actual, strains_desired)
        assert_array_almost_equal(stresses_actual, stresses_desired)

        # 8 Tri3-Elements
        here = dirname(abspath(__file__))
        mesh_input = join(here, 'meshes', 'gmsh_8_tris.msh')
        mesh_reader = GmshAsciiMeshReader(mesh_input)
        converter = AmfeMeshConverter()
        mesh_reader.parse(converter)
        mesh = converter.return_mesh()
        amp = 1.0
        my_comp = StructuralComponent(mesh)

        my_material = KirchhoffMaterial(E=30e6, nu=0.0, plane_stress=True)
        my_comp.assign_material(my_material, ['volume'], 'S')

        q = np.array([0.15, 0, 0.1, 0, 0.1, 0, 0.05, 0, 0, 0, 0.2, 0, 0.2, 0, 0, 0, 0, 0])
        dq = np.zeros(my_comp.constraints.no_of_dofs_unconstrained)
        strains_actual, stresses_actual = my_comp.strains_and_stresses(q, dq, 0.0)

        strains_desired = np.array([[1.05e-1, 0, 0, 0, 0, 0],
                                    [1.05e-1, 0, 0, 0, 0, 0],
                                    [1.05e-1, 0, 0, 0, 0, 0],
                                    [1.05e-1, 0, 0, 0, 0, 0],
                                    [1.05e-1, 0, 0, 0, 0, 0],
                                    [1.05e-1, 0, 0, 0, 0, 0],
                                    [1.05e-1, 0, 0, 0, 0, 0],
                                    [1.05e-1, 0, 0, 0, 0, 0]])

        stresses_desired = np.array([[3.15e6, 0, 0, 0, 0, 0],
                                     [3.15e6, 0, 0, 0, 0, 0],
                                     [3.15e6, 0, 0, 0, 0, 0],
                                     [3.15e6, 0, 0, 0, 0, 0],
                                     [3.15e6, 0, 0, 0, 0, 0],
                                     [3.15e6, 0, 0, 0, 0, 0],
                                     [3.15e6, 0, 0, 0, 0, 0],
                                     [3.15e6, 0, 0, 0, 0, 0]])

        assert_array_almost_equal(strains_actual, strains_desired)
        assert_array_almost_equal(stresses_actual, stresses_desired, 5)
