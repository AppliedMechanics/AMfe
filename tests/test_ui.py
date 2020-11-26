# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from unittest import TestCase
from os.path import join, dirname, abspath
import numpy as np
from numpy.testing import assert_array_equal

from amfe.ui import *
from amfe.component import MeshComponent
from amfe.mesh import Mesh
from amfe.material import Material
from amfe.forces import *
from amfe.neumann.structural_neumann import FixedDirectionNeumann, NormalFollowingNeumann
from amfe.solver.translators import create_constrained_mechanical_system_from_component
from amfe.solver import AmfeSolution


class TestUi(TestCase):
    def setUp(self):
        self.here = dirname(abspath(__file__))
        self.input_file_1 = join(self.here, 'meshes', '2_partitions_2quad_mesh.msh')
        self.input_file_2 = join(self.here, 'meshes', 'gid_json_4_tets.json')

        class DummyMaterial:
            def __init__(self, name):
                self.name = name

        self.testmaterial = DummyMaterial('steel')

    def tearDown(self):
        pass

    def test_import_mesh_from_file(self):
        input_file_3 = join(self.here, 'meshes', 'not_a_mesh_for_testing.txt')
        testmesh_1 = import_mesh_from_file(self.input_file_1)
        testmesh_2 = import_mesh_from_file(self.input_file_2)
        # check if raises error for no valid mesh file
        self.assertRaises(ValueError, import_mesh_from_file, input_file_3)
        # check for correct import
        self.assertIsInstance(testmesh_1, Mesh)
        self.assertIsInstance(testmesh_2, Mesh)
        self.assertEqual(testmesh_1.no_of_elements, 2)
        self.assertEqual(testmesh_2.no_of_elements, 4)

    def test_create_structural_component(self):
        testmesh_1 = import_mesh_from_file(self.input_file_1)
        testcomponent = create_structural_component(testmesh_1)
        self.assertIsInstance(testcomponent, MeshComponent)

    def test_create_material(self):
        testmaterial_1 = create_material('Kirchhoff', E=210E9, nu=0.3, rho=7.86E3,
                                         plane_stress=True, thickness=0.1)
        testmaterial_2 = create_material('Kirchhoff')
        testmaterial_3 = create_material('MooneyRivlin', A10=0.3E3, A01=0.2E3, kappa=2E5, rho=0.95E3,
                                         plane_stress=False, thickness=2.0)
        testmaterial_4 = create_material('MooneyRivlin')
        self.assertIsInstance(testmaterial_1, Material)
        self.assertIsInstance(testmaterial_2, Material)
        self.assertIsInstance(testmaterial_3, Material)
        self.assertIsInstance(testmaterial_4, Material)
        self.assertRaises(ValueError, create_material, 'Not_a_material', E=210E9, nu=0.3, rho=7.86E3,
                          plane_stress=True, thickness=0.1)

    def test_assign_material_by_group(self):
        testmesh = import_mesh_from_file(self.input_file_1)
        testcomponent = create_structural_component(testmesh)
        # test for error-message when using wrong group-name
        self.assertRaises(ValueError, assign_material_by_group, testcomponent, self.testmaterial, 'not_a_group')
        # check if material successfully assigned
        assign_material_by_group(testcomponent, self.testmaterial, '1')
        material = testcomponent.get_materials()
        self.assertEqual(material[0], self.testmaterial)

    def test_assign_material_by_elementids(self):
        testmesh = import_mesh_from_file(self.input_file_1)
        testcomponent = create_structural_component(testmesh)
        eleids1 = np.array([1, 2], dtype=int)
        assign_material_by_elementids(testcomponent, self.testmaterial, eleids1)
        # check if material successfully assigned
        eleids1_real = testcomponent.get_elementids_by_materials(self.testmaterial)
        assert_array_equal(eleids1, eleids1_real)

    def test_set_dirichlet_by_group(self):
        testmesh = import_mesh_from_file(self.input_file_1)
        testcomponent = create_structural_component(testmesh)
        assign_material_by_group(testcomponent, self.testmaterial, '1')
        assign_material_by_group(testcomponent, self.testmaterial, '2')
        assign_material_by_group(testcomponent, self.testmaterial, '3')

        # assign Dirichlet_ux and test
        group_name = '1'
        direction = 'ux'
        constraint_name = 'Dirichlet_ux'
        set_dirichlet_by_group(testcomponent, group_name, direction, constraint_name)
        self.assertEqual(testcomponent.constraints.no_of_constraints, 2)

        # assign Dirichlet_uy and test
        group_name = '2'
        direction = 'uy'
        constraint_name = 'Dirichlet_uy'
        set_dirichlet_by_group(testcomponent, group_name, direction, constraint_name)
        self.assertEqual(testcomponent.constraints.no_of_constraints, 4)

    def test_set_dirichlet_by_nodeids(self):
        testmesh = import_mesh_from_file(self.input_file_1)
        testcomponent = create_structural_component(testmesh)
        assign_material_by_group(testcomponent, self.testmaterial, '1')
        assign_material_by_group(testcomponent, self.testmaterial, '2')
        assign_material_by_group(testcomponent, self.testmaterial, '3')

        # assign Dirichlet_ux
        nodeids = np.array([1, 2, 3])
        direction = 'ux'
        constraint_name = 'Dirichlet_ux'
        set_dirichlet_by_nodeids(testcomponent, nodeids, direction, constraint_name)
        self.assertEqual(testcomponent.constraints.no_of_constraints, 3)

        # assign Dirichlet_uy
        nodeids = np.array([4, 5, 6])
        direction = 'ux'
        constraint_name = 'Dirichlet_uy'
        set_dirichlet_by_nodeids(testcomponent, nodeids, direction, constraint_name)
        self.assertEqual(testcomponent.constraints.no_of_constraints, 6)

    def test_set_neumann_by_group(self):
        testmesh = import_mesh_from_file(self.input_file_1)
        testcomponent = create_structural_component(testmesh)
        assign_material_by_group(testcomponent, self.testmaterial, '1')
        assign_material_by_group(testcomponent, self.testmaterial, '2')
        assign_material_by_group(testcomponent, self.testmaterial, '3')

        # assign Neumann-condition
        group_name = '1'
        direction_vector = np.array([0, 1])
        neumann_name = 'Neumann0'
        force = constant_force(1.0)
        set_neumann_by_group(testcomponent, group_name, direction_vector, neumann_name=neumann_name, f=force)
        # test
        neumann_obj, fk_mesh, fk_mapping = testcomponent.neumann.get_ele_obj_fk_mesh_and_fk_mapping()
        self.assertIsInstance(neumann_obj[0], FixedDirectionNeumann)

    def test_set_neumann_by_elementids(self):
        testmesh = import_mesh_from_file(self.input_file_1)
        testcomponent = create_structural_component(testmesh)
        assign_material_by_group(testcomponent, self.testmaterial, '1')
        assign_material_by_group(testcomponent, self.testmaterial, '2')
        assign_material_by_group(testcomponent, self.testmaterial, '3')

        # assign Neumann-condition
        eleids1 = np.array([1, 2], dtype=int)
        direction_vector = np.array([0, 1])
        neumann_name = 'Neumann0'
        force = constant_force(1.0)
        set_neumann_by_elementids(testcomponent, eleids1, direction_vector, neumann_name=neumann_name, f=force)

        # assign Neumann-condition
        direction_vector = 'normal'
        neumann_name = 'Neumann1'
        force = constant_force(1.0)
        set_neumann_by_elementids(testcomponent, eleids1, direction_vector, following=True, neumann_name=neumann_name,
                                  f=force)

        # test
        neumann_obj, fk_mesh, fk_mapping = testcomponent.neumann.get_ele_obj_fk_mesh_and_fk_mapping()
        self.assertIsInstance(neumann_obj[0], FixedDirectionNeumann)
        self.assertIsInstance(neumann_obj[2], NormalFollowingNeumann)

    def test_solve_modes(self):
        testmesh_1 = import_mesh_from_file(self.input_file_1)
        testcomponent = create_structural_component(testmesh_1)
        testmaterial_1 = create_material('Kirchhoff', E=210E9, nu=0.3, rho=7.86E3,
                                         plane_stress=True, thickness=0.1)
        assign_material_by_group(testcomponent, testmaterial_1, '3')
        testsystem, testformulation = create_constrained_mechanical_system_from_component(testcomponent)
        modes = solve_modes(testsystem, testformulation, 10)
        self.assertIsInstance(modes, AmfeSolution)
        self.assertEqual(len(modes.t), 10)
        self.assertEqual(len(modes.ddq), 10)
        self.assertEqual(len(modes.dq), 10)
        self.assertEqual(len(modes.q), 10)
