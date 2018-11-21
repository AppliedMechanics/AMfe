"""Test Routine for component"""


from unittest import TestCase
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from amfe.component.tree_manager import *
from amfe.component.structural_component import StructuralComponent


class StructuralComponentTest(TestCase):
    def setUp(self):
        class DummyConstraint:
            def __init__(self):
                self.no_of_constrained_dofs = 2

            def unconstrain_u(self, u_constr):
                return np.array([0, 0, 0])

            def constrain_matrix(self,K_unconstr):
                return K_unconstr[0:2, 0:2]

        class DummyAssembly:
            def __init__(self):
                pass

            def assemble_k_and_f(self, u, t):
                K_unconstr=np.array([[10, -5, 0], [-5, 10, -5], [0, -5, 10]])
                f_unsonstr=np.array([2, 0, 0])
                return K_unconstr, f_unsonstr

        class DummyMesh:
            def __init__(self,dimension):
                self.dimension = dimension
                self.nodes = np.empty((0, dimension), dtype=float)
                self.connectivity = list()
                self.el_df = pd.DataFrame(columns=['shape', 'is_boundary', 'connectivity_idx'])

            @property
            def no_of_elements(self):
                return 0

        self.mesh = DummyMesh(2)
        self.assembly = DummyAssembly()

        self.structComp = []
        comp = StructuralComponent(self.mesh)
        comp._constraints = DummyConstraint()
        comp._assembly = DummyAssembly()
        self.structComp.append(comp)

    def tearDown(self):
        pass

    def test_initialization(self):
        pass

    def test_k(self):
        desiredK=np.array([[10, -5], [-5, 10]])
        assert_array_equal(self.structComp[0].K(), desiredK)
        
    def test_get_mat(self):
        desiredK = np.array([[10, -5], [-5, 10]])
        assert_array_equal(self.structComp[0].get_mat('K'), desiredK)


class ComponentCompositeTest(TestCase):
    def setUp(self):
        class DummyConstraint:
            def __init__(self):
                self.no_of_constrained_dofs = 2

            def unconstrain_u(self, u_constr):
                return np.array([0, 0, 0])

            def constrain_matrix(self, K_unconstr):
                return K_unconstr[0:2, 0:2]

        class DummyAssembly:
            def __init__(self):
                pass

            def assemble_k_and_f(self, u, t):
                K_unconstr=np.array([[10, -5, 0], [-5, 10, -5], [0, -5, 10]])
                f_unsonstr=np.array([2, 0, 0])
                return K_unconstr, f_unsonstr

        class DummyMesh:
            def __init__(self, dimension):
                self.dimension = dimension
                self.nodes = np.empty((0, dimension), dtype=float)
                self.connectivity = list()
                self.el_df = pd.DataFrame(columns=['shape', 'is_boundary', 'connectivity_idx'])

            @property
            def no_of_elements(self):
                return 0

        class DummyLeaf:
            def __init__(self):
                pass

            def get_local_component_id(self, leaf_id, composite_layer):
                return leaf_id

        self.mesh = DummyMesh(2)
        self.assembly = DummyAssembly()
        self.constraints = DummyConstraint()
        self.leaf_paths = DummyLeaf()

        self.TestComponent = []
        comp = StructuralComponent(self.mesh)
        comp._constraints = self.constraints
        comp._assembly = self.assembly

        self.TestComponent.append(comp)
        self.CompComposite = ComponentComposite(self.leaf_paths, self.TestComponent)

    def tearDown(self):
        pass

    def test_add_components(self):
        # Test normal component
        prev_N_components = self.CompComposite.no_of_components

        TestComponent2 = deepcopy(self.TestComponent)
        self.CompComposite.add_component(TestComponent2[0])

        self.assertEqual(self.CompComposite.no_of_components, 2)
        self.assertTrue(self.TestComponent[0] in self.CompComposite.components)
        self.assertTrue(TestComponent2[0] in self.CompComposite.components)

        #Test composite component
        prev_N_components = self.CompComposite.no_of_components
        TestComposite = ComponentComposite(self.leaf_paths)
        TestComposite.add_component(self.TestComponent + TestComponent2)

        self.CompComposite.add_component(TestComposite)

        self.assertEqual(self.CompComposite.no_of_components, prev_N_components+1)
        self.assertTrue(TestComposite in self.CompComposite.components)

    def test_delete_component(self):
        TestComponent2 = deepcopy(self.TestComponent)
        self.CompComposite.add_component(TestComponent2)
        TestComposite = ComponentComposite(self.leaf_paths, self.TestComponent)
        TestComposite.add_component(TestComponent2)
        self.CompComposite.add_component(TestComposite)

        prev_N_components = self.CompComposite.no_of_components

        self.CompComposite.delete_component(1)

        self.assertEqual(self.CompComposite.no_of_components, prev_N_components-1)
        self.assertTrue(TestComponent2 not in self.CompComposite.components)


class TreeBuilderTest(TestCase):
    def setUp(self):
        class DummyConstraint:
            def __init__(self):
                self.no_of_constrained_dofs = 2

            def unconstrain_u(self, u_constr):
                return np.array([0, 0, 0])

            def constrain_matrix(self, K_unconstr):
                return K_unconstr[0:2, 0:2]

        class DummyAssembly:
            def __init__(self):
                pass

            def assemble_k_and_f(self, u, t):
                K_unconstr=np.array([[10, -5, 0], [-5, 10, -5], [0, -5, 10]])
                f_unsonstr=np.array([2, 0, 0])
                return K_unconstr, f_unsonstr

        class DummyMesh:
            def __init__(self, dimension):
                self.dimension = dimension
                self.nodes = np.empty((0, dimension), dtype=float)
                self.connectivity = list()
                self.el_df = pd.DataFrame(columns=['shape', 'is_boundary', 'connectivity_idx'])

            @property
            def no_of_elements(self):
                return 0

        self.mesh = DummyMesh(2)
        self.assembly = DummyAssembly()
        self.constraints = DummyConstraint()

        self.TestComponent = []
        comp = StructuralComponent(self.mesh)
        comp._constraints = self.constraints
        comp._assembly = self.assembly

        self.TestComponent.append(comp)

        self.tree = TreeBuilder()

    def tearDown(self):
        pass

    def test_add(self):
        # Test normal component
        desiredLeafPaths = {0: [0],
                          1: [1]}

        TestComponent2 = deepcopy(self.TestComponent)
        self.tree.add(self.TestComponent + TestComponent2)

        self.assertEqual(self.tree.root_composite.no_of_components, 2)
        self.assertTrue(self.tree.root_composite.components[0] == self.TestComponent[0])
        self.assertTrue(TestComponent2[0] in self.tree.root_composite.components)
        self.assertEqual(desiredLeafPaths, self.tree.leaf_paths.leaves)

        #Test composite component
        TestTree = TreeBuilder()

        TestTree.add(self.TestComponent + TestComponent2)
        TestTree2 = deepcopy(TestTree)
        TestTree.add(TestTree2.root_composite)

        desiredLeafPaths = {0: [0],
                            1: [1],
                            2: [2, 0],
                            3: [2, 1]}

        self.assertEqual(desiredLeafPaths, TestTree.leaf_paths.leaves)

        self.tree.add(TestTree.root_composite)

        desiredLeafPaths = {0: [0],
                           1: [1],
                           2: [2, 0],
                           3: [2, 1],
                           4: [2, 2, 0],
                           5: [2, 2, 1]}

        self.assertEqual(self.tree.root_composite.no_of_components, 3)
        self.assertTrue(TestTree.root_composite in self.tree.root_composite.components)
        self.assertEqual(desiredLeafPaths, self.tree.leaf_paths.leaves)

    def test_delete(self):
        TestComponent2 = deepcopy(self.TestComponent)
        self.tree.add(self.TestComponent + TestComponent2)
        TestTree = TreeBuilder()
        TestTree.add(self.TestComponent + TestComponent2)
        TestTree2 = deepcopy(TestTree)
        TestTree.add(TestTree2.root_composite)
        self.tree.add(TestTree.root_composite)
        self.tree.add(self.TestComponent)

        self.tree.delete_leafs(1)

        desiredLeafPaths = {0: [0],
                           2: [1, 0],
                           3: [1, 1],
                           4: [1, 2, 0],
                           5: [1, 2, 1],
                           6: [2]}

        self.assertEqual(self.tree.root_composite.no_of_components, 3)
        self.assertTrue(TestComponent2 not in self.tree.root_composite.components)
        self.assertEqual(desiredLeafPaths, self.tree.leaf_paths.leaves)
        self.assertEqual(desiredLeafPaths, self.tree.root_composite.components[1].leaf_paths.leaves)

        self.tree.delete_component([1, 2],0)

        desiredLeafPaths = {0: [0],
                           2: [1, 0],
                           3: [1, 1],
                           5: [1, 2, 0],
                           6: [2]}

        self.assertTrue(TestComponent2 not in self.tree.root_composite.components[1].components)
        self.assertEqual(desiredLeafPaths, self.tree.leaf_paths.leaves)

        self.tree.delete_component([1],0)

        desiredLeafPaths = {0: [0],
                           3: [1, 0],
                           5: [1, 1, 0],
                           6: [2]}

        self.assertEqual(desiredLeafPaths, self.tree.leaf_paths.leaves)

        self.tree.delete_component([1],0)

        desiredLeafPaths = {0: [0],
                           5: [1, 0, 0],
                           6: [2]}

        self.assertEqual(desiredLeafPaths, self.tree.leaf_paths.leaves)

        self.tree.delete_component([],1)

        desiredLeafPaths = {0: [0],
                            6: [1]}

        self.assertEqual(desiredLeafPaths, self.tree.leaf_paths.leaves)




