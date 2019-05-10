"""Test Routine for tree-builder and leaf-paths"""

from unittest import TestCase
from copy import deepcopy

from amfe.component.tree_manager import *
from amfe.component import ComponentBase


class DummyComponent(ComponentBase):
    def __init__(self):
        pass


class DummyConnector:
    def __init__(self):
        self.dofs_map = []

    def set_dofs_mapping_local2global(self, dofs_map):
        self.dofs_map = dofs_map

    def apply_compatibility_constraint(self, master_id, master_comp, slave_id, slave_comp):
        pass


class TreeBuilderTest(TestCase):
    def setUp(self):
        class DummySeparator:
            def __init__(self):
                pass

            def separate_partitioned_component(self, component):
                new_components = [deepcopy(component), deepcopy(component), deepcopy(component)]
                dofs_map = []
                new_ids = [0, 1, 2]
                return new_ids, new_components, dofs_map

        self.TestComponent = []
        comp = DummyComponent()

        self.TestComponent.append(comp)

        self.tree = TreeBuilder(separator=DummySeparator())
        self.tree.root_composite = ComponentComposite()
        self.tree.root_composite.connector = DummyConnector()

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

    def test_delete_leafs_and_component(self):
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

    def test_separate_partitioned_component_by_leafid(self):
        # Simple separation of single component at root-composite
        self.tree.add([0], self.TestComponent)

        self.assertTrue(len(self.tree.root_composite.components) == 1)

        self.tree.separate_partitioned_component_by_leafid(0)

        self.assertTrue(len(self.tree.root_composite.components) == 3)
        self.assertTrue(self.TestComponent[0] not in self.tree.root_composite.components.values())

        for icomp in self.tree.root_composite.components.values():
            self.assertTrue(isinstance(icomp, DummyComponent))


