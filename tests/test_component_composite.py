"""Test Routine for component-composite"""


from unittest import TestCase
from copy import deepcopy
import numpy as np
import pandas as pd
from amfe.component.structural_component import StructuralComponent
from amfe.component.component_composite import ComponentComposite


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

        comp = StructuralComponent(self.mesh)
        comp._constraints = self.constraints
        comp._assembly = self.assembly

        self.TestComponent=comp
        self.CompComposite = ComponentComposite(self.TestComponent)

    def tearDown(self):
        pass

    def test_add_component(self):
        # Test normal component
        prev_N_components = self.CompComposite.no_of_components

        TestComponent2 = deepcopy(self.TestComponent)
        self.CompComposite.add_component(1, TestComponent2)

        self.assertEqual(self.CompComposite.no_of_components, 2)
        self.assertTrue(self.TestComponent in self.CompComposite.components.values())
        self.assertTrue(TestComponent2 in self.CompComposite.components.values())

        #Test composite component
        prev_N_components = self.CompComposite.no_of_components
        TestComposite = ComponentComposite()
        TestComposite.add_component(0, self.TestComponent)
        TestComposite.add_component(1, TestComponent2)

        self.CompComposite.add_component(2, TestComposite)

        self.assertEqual(self.CompComposite.no_of_components, prev_N_components+1)
        self.assertTrue(TestComposite in self.CompComposite.components.values())

        self.assertRaises(TypeError, lambda: self.CompComposite.add_component(3, [self.TestComponent]))
        self.assertRaises(ValueError, lambda: self.CompComposite.add_component(1, self.TestComponent))
        
    def test_replace_component(self):
        TestComponent2 = deepcopy(self.TestComponent)
        self.CompComposite.add_component(1, TestComponent2)
        
        TestComponent3 = deepcopy(self.TestComponent)
        self.CompComposite.replace_component(1, TestComponent3)
        
        self.assertEqual(self.CompComposite.no_of_components, 2)
        self.assertTrue(self.TestComponent == self.CompComposite.components[0])
        self.assertTrue(TestComponent3 == self.CompComposite.components[1])

        self.CompComposite.add_component(2, TestComponent2)
        TestComposite = ComponentComposite([self.TestComponent, TestComponent2])
        
        self.CompComposite.replace_component(1, TestComposite)
        self.assertEqual(self.CompComposite.no_of_components, 3)
        self.assertTrue(self.TestComponent == self.CompComposite.components[0])
        self.assertTrue(TestComponent2 == self.CompComposite.components[2])
        self.assertTrue(TestComposite == self.CompComposite.components[1])
        self.assertTrue(self.TestComponent == self.CompComposite.components[1].components[0])
        self.assertTrue(TestComponent2 == self.CompComposite.components[1].components[1])

    def test_delete_component(self):
        TestComponent2 = deepcopy(self.TestComponent)
        self.CompComposite.add_component(1, TestComponent2)
        TestComposite = ComponentComposite(self.TestComponent)
        TestComposite.add_component(3, TestComponent2)
        self.CompComposite.add_component(2, TestComposite)

        prev_N_components = self.CompComposite.no_of_components

        self.CompComposite.delete_component(1)

        self.assertEqual(self.CompComposite.no_of_components, prev_N_components-1)
        self.assertTrue(TestComponent2 not in self.CompComposite.components.values())




