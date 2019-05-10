"""Test Routine for component"""


from unittest import TestCase
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from amfe.component.tree_manager import *
from amfe.component.structural_component import StructuralComponent
from copy import deepcopy


class StructuralComponentTest(TestCase):
    def setUp(self):
        class DummyConstraint:
            def __init__(self):
                self.no_of_constrained_dofs = 2

            def unconstrain_vector(self, u_free, u_constr=None):
                return np.array([0, 0, 0])

            def constrain_matrix(self, K_unconstr, t=0):
                return K_unconstr[0:2, 0:2]

        class DummyAssembly:
            def __init__(self):
                pass

            def assemble_k_and_f(self, nodes, ele_objects, connectivities, elements2dofs,
                                 dofvalues=None, t=0., C_csr=None, f_glob=None):
                if C_csr is None:
                    C_csr = np.array([[10, -5, 0], [-5, 10, -5], [0, -5, 10]])
                else:
                    C_csr[:, :] = np.array([[10, -5, 0], [-5, 10, -5], [0, -5, 10]])
                if f_glob is None:
                    f_glob = np.array([2, 0, 0])
                else:
                    f_glob[:] = np.array([2, 0, 0])
                return C_csr, f_glob

            def preallocate(self, no_of_dofs, elements2dof):
                return np.zeros((3, 3))

        class DummyMesh:
            def __init__(self, dimension):
                self.dimension = dimension
                self.nodes_df = pd.DataFrame({'x': [0.0, 1.0, 0.0], 'y': [0.0, 0.0, 1.0]})
                self.connectivity = None
                self.el_df = pd.DataFrame({'shape': ['Tri3'], 'is_boundary': [False],
                                           'connectivity': [np.array([1, 2, 3], dtype=int)]})

            @property
            def no_of_elements(self):
                return 0
            
            @property
            def nodes(self):
                return np.zeros(3)
            
            def get_nodeidxs_by_all(self):
                return np.arange(0, 3)
            
            def get_nodeids_by_nodeidxs(self, nodeidxs):
                return np.arange(0, 3)

            def get_iconnectivity_by_elementids(self, elementids):
                return None
                
        class DummyMapping:
            def __init__(self):
                pass
            
            @property
            def no_of_dofs(self):
                return 3
            
            def get_dofs_by_nodeids(self, nodeids):
                return np.arange(0, 3)
            
            def get_dofs_by_ids(self, ids):
                return None


        self.mesh = DummyMesh(2)
        self.assembly = DummyAssembly()

        self.structComp = []
        comp = StructuralComponent(self.mesh)
        comp._mapping = DummyMapping()
        comp._constraints = DummyConstraint()
        comp._assembly = DummyAssembly()
        comp._C_csr = comp._assembly.preallocate(2, None)
        comp._f_glob_int = np.zeros(comp._C_csr.shape[0])
        self.structComp.append(comp)

    def tearDown(self):
        pass

    def test_initialization(self):
        pass

    def test_k(self):
        desiredK=np.array([[10, -5, 0], [-5, 10, -5], [0, -5, 10]])
        q = dq = np.zeros(3)
        t = 0.0
        assert_array_equal(self.structComp[0].K(q, dq, t), desiredK)

    def test_get_mat(self):
        desiredK = np.array([[10, -5, 0], [-5, 10, -5], [0, -5, 10]])
        q = dq = np.zeros(3)
        t = 0.0
        assert_array_equal(self.structComp[0].get_mat('K', q, dq, t), desiredK)


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
                self.connectivity = None
                self.el_df = pd.DataFrame(columns=['shape', 'is_boundary', 'connectivity'])

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
        self.CompComposite = ComponentComposite(self.TestComponent)

    def tearDown(self):
        pass

    def test_add_components(self):
        # Test normal component
        prev_N_components = self.CompComposite.no_of_components

        TestComponent2 = deepcopy(self.TestComponent)
        self.CompComposite.add_component(1, TestComponent2[0])

        self.assertEqual(self.CompComposite.no_of_components, 2)
        self.assertTrue(self.TestComponent[0] in self.CompComposite.components.values())
        self.assertTrue(TestComponent2[0] in self.CompComposite.components.values())

        #Test composite component
        prev_N_components = self.CompComposite.no_of_components
        TestComposite = ComponentComposite()
        TestComposite.add_component(0, self.TestComponent[0])
        TestComposite.add_component(1, TestComponent2[0])

        self.CompComposite.add_component(2, TestComposite)

        self.assertEqual(self.CompComposite.no_of_components, prev_N_components+1)
        self.assertTrue(TestComposite in self.CompComposite.components.values())

        self.assertRaises(TypeError, lambda : self.CompComposite.add_component(3, self.TestComponent))
        self.assertRaises(ValueError, lambda : self.CompComposite.add_component(1, self.TestComponent[0]))

    def test_delete_component(self):
        TestComponent2 = deepcopy(self.TestComponent)
        self.CompComposite.add_component(1, TestComponent2[0])
        TestComposite = ComponentComposite(self.TestComponent)
        TestComposite.add_component(3, TestComponent2[0])
        self.CompComposite.add_component(2, TestComposite)

        prev_N_components = self.CompComposite.no_of_components

        self.CompComposite.delete_component(1)

        self.assertEqual(self.CompComposite.no_of_components, prev_N_components-1)
        self.assertTrue(TestComponent2 not in self.CompComposite.components.values())


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
                self.connectivity = None
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
        self.tree.add([0, 1], self.TestComponent + TestComponent2)

        self.assertEqual(self.tree.root_composite.no_of_components, 2)
        self.assertTrue(self.tree.root_composite.components[0] == self.TestComponent[0])
        self.assertTrue(TestComponent2[0] in self.tree.root_composite.components.values())
        self.assertEqual(desiredLeafPaths, self.tree.leaf_paths.leaves)

        #Test composite component
        TestTree = TreeBuilder()

        TestTree.add([0, 1], self.TestComponent + TestComponent2)
        TestTree2 = deepcopy(TestTree)
        TestTree.add([2], TestTree2.root_composite)

        desiredLeafPaths = {0: [0],
                            1: [1],
                            2: [2, 0],
                            3: [2, 1]}

        self.assertEqual(desiredLeafPaths, TestTree.leaf_paths.leaves)

        self.tree.add([2], TestTree.root_composite)

        desiredLeafPaths = {0: [0],
                           1: [1],
                           2: [2, 0],
                           3: [2, 1],
                           4: [2, 2, 0],
                           5: [2, 2, 1]}

        self.assertEqual(self.tree.root_composite.no_of_components, 3)
        self.assertTrue(TestTree.root_composite in self.tree.root_composite.components.values())
        self.assertEqual(desiredLeafPaths, self.tree.leaf_paths.leaves)

    def test_delete(self):
        TestComponent2 = deepcopy(self.TestComponent)
        self.tree.add([0, 1], self.TestComponent + TestComponent2)
        TestTree = TreeBuilder()
        TestTree.add([0, 1], self.TestComponent + TestComponent2)
        TestTree2 = deepcopy(TestTree)
        TestTree.add([2], TestTree2.root_composite)
        self.tree.add([2], TestTree.root_composite)
        self.tree.add([3], self.TestComponent)

        self.tree.delete_leafs(1)

        desiredLeafPaths = {0: [0],
                           2: [2, 0],
                           3: [2, 1],
                           4: [2, 2, 0],
                           5: [2, 2, 1],
                           6: [3]}

        self.assertEqual(self.tree.root_composite.no_of_components, 3)
        self.assertTrue(TestComponent2 not in self.tree.root_composite.components.values())
        self.assertEqual(desiredLeafPaths, self.tree.leaf_paths.leaves)

        self.tree.delete_component([2, 2], 0)

        desiredLeafPaths = {0: [0],
                           2: [2, 0],
                           3: [2, 1],
                           5: [2, 2, 1],
                           6: [3]}

        self.assertTrue(TestComponent2 not in self.tree.root_composite.components[2].components.values())
        self.assertEqual(desiredLeafPaths, self.tree.leaf_paths.leaves)

        self.tree.delete_component([2], 0)

        desiredLeafPaths = {0: [0],
                           3: [2, 1],
                           5: [2, 2, 1],
                           6: [3]}

        self.assertEqual(desiredLeafPaths, self.tree.leaf_paths.leaves)

        self.tree.delete_component([2], 1)

        desiredLeafPaths = {0: [0],
                           5: [2, 2, 1],
                           6: [3]}

        self.assertEqual(desiredLeafPaths, self.tree.leaf_paths.leaves)

        self.tree.delete_component([], 2)

        desiredLeafPaths = {0: [0],
                            6: [3]}

        self.assertEqual(desiredLeafPaths, self.tree.leaf_paths.leaves)




