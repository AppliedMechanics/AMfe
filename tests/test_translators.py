"""
Test for testing the translator-module
"""

from unittest import TestCase
import numpy as np
from amfe.solver.translators import *
from numpy.testing import assert_array_equal, assert_equal, assert_allclose


class TranslatorsTest(TestCase):
    def setUp(self):
        class DummyMapping:
            def __init__(self):
                return

            @property
            def no_of_dofs(self):
                return 3

        dummy_mapping = DummyMapping()

        class DummyStructuralComponent:
            def __init__(self):
                pass

            @property
            def X(self):
                return np.array([0.1, 0.2, 0.3])

            def K(self, q, dq, t):
                return np.array([[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]]) * np.linalg.norm(q)

            def M(self, q, dq, t):
                return np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]) * np.linalg.norm(q)

            def D(self, q, dq, t):
                return np.array([[0.3, 0, 0], [0, 0.3, 0], [0, 0.3, 0]]) * np.linalg.norm(q)

            def f_int(self, q, dq, t):
                return self.K(q, dq, t) @ q

            def K_and_f_int(self, q, dq, t):
                return self.K(q, dq, t), self.f_int(q, dq, t)

            def f_ext(self, q, dq, t):
                return np.array([0., 0., 1])

            @property
            def mapping(self):
                return dummy_mapping
    
        self.structural_component = DummyStructuralComponent()
        
    def tearDown(self):
        pass
    
    def test_mechanical_translator_base(self):
        translator = create_mechanical_system_from_structural_component(self.structural_component)
        
        u = np.array([1, 0, 0], dtype=float)
        du = np.zeros_like(u)
        t = 0.0

        K_desired = np.array([[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]])
        M_desired = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
        D_desired = np.array([[0.3, 0, 0], [0, 0.3, 0], [0, 0.3, 0]])
        f_int_desired = K_desired @ u
        f_ext_desired = np.array([0., 0., 1])

        assert_array_equal(K_desired, translator.K(u, du, t))
        assert_array_equal(D_desired, translator.D(u, du, t))
        assert_array_equal(M_desired, translator.M(u, du, t))
        assert_array_equal(f_ext_desired, translator.f_ext(u, du, t))
        assert_array_equal(f_int_desired, translator.f_int(u, du, t))

    def test_full_linear_mechanical_translator(self):
        dim = self.structural_component.mapping.no_of_dofs
        M = self.structural_component.M
        D = self.structural_component.D
        K = self.structural_component.K
        f_ext = self.structural_component.f_ext

        self.assertRaises(ValueError, lambda: MechanicalSystem(dim, M, D, K, f_ext))

        translator = MechanicalSystem(dim, M, D, K, f_ext, None, ('all',))
        self.assertTrue(translator.system_is_linear)

        u = np.array([0.05, 0.1, 0.15])
        du = np.zeros_like(u)
        t = 0.0
        assert_array_equal(translator.K(u * 0.1, du, t), translator.K(u, du, t))
        assert_array_equal(translator.D(u * 0.1, du, t), translator.D(u, du, t))
        assert_array_equal(translator.M(u * 0.1, du, t), translator.M(u, du, t))

    def test_partial_linear_mechanical_translator(self):
        dim = self.structural_component.mapping.no_of_dofs
        M = self.structural_component.M
        D = self.structural_component.D
        K = self.structural_component.K
        f_ext = self.structural_component.f_ext
        f_int = self.structural_component.f_int

        translator = MechanicalSystem(dim, M, D, K, f_ext, f_int, ('M', 'K'))

        u = np.array([0.05, 0.1, 0.15])
        du = np.zeros_like(u)
        t = 0.0
        assert_allclose(translator.K(u + 0.1, du, t), translator.K(u, du, t))
        assert_equal(np.any(np.not_equal( translator.D(u * 0.1, du, t), translator.D(u, du, t) )), True)
        assert_allclose(translator.M(u + 0.1, du, t), translator.M(u, du, t))

        self.assertFalse(translator.system_is_linear)

    def test_custom_matrices(self):
        M_cust = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
        K_cust = np.array([[2, 0.35, 0], [0.35, 2, 0.35], [0, 0.35, 1]])
        f_ext_cust = np.array([0., 0., 1.3])
        dim = self.structural_component.mapping.no_of_dofs
        D = self.structural_component.D
        f_int = self.structural_component.f_int

        # Full linear system with some custom matrices set
        translator = MechanicalSystem(dim, M_cust, D, K_cust, f_ext_cust, f_int, ('all',))

        self.assertFalse(translator.system_is_linear)

        u = np.array([0.05, 0.1, 0.15])
        du = np.zeros_like(u)
        t = 0.0
        assert_array_equal(translator.K(u * 0.1, du, t), translator.K(u, du, t))
        assert_array_equal(translator.D(u * 0.1, du, t), translator.D(u, du, t))
        assert_array_equal(translator.M(u * 0.1, du, t), translator.M(u, du, t))
        assert_array_equal(translator.f_ext(u * 0.1, du, t), translator.f_ext(u, du, t))

        assert_array_equal(translator.K(u * 0.1, du, t), K_cust)
        assert_array_equal(translator.M(u * 0.1, du, t), M_cust)
        assert_array_equal(translator.f_ext(u * 0.1, du, t), f_ext_cust)

        # Nonlinear system with custom linear M, K, f_ext and linear calculated D
        translator = MechanicalSystem(dim, M_cust, D, K_cust, f_ext_cust, f_int, ('D',))

        u = np.array([0.05, 0.1, 0.15])
        du = np.zeros_like(u)
        t = 0.0
        assert_array_equal(translator.K(u * 0.1, du, t), translator.K(u, du, t))
        assert_array_equal(translator.D(u * 0.1, du, t), translator.D(u, du, t))
        assert_array_equal(translator.M(u * 0.1, du, t), translator.M(u, du, t))
        assert_array_equal(translator.f_ext(u * 0.1, du, t), translator.f_ext(u, du, t))

        assert_array_equal(translator.K(u * 0.1, du, t), K_cust)
        assert_array_equal(translator.M(u * 0.1, du, t), M_cust)
        assert_array_equal(translator.f_ext(u * 0.1, du, t), f_ext_cust)

        # Nonlinear system with custom linear M, K, f_ext
        translator = MechanicalSystem(dim, M_cust, D, K_cust, f_ext_cust, f_int)

        u = np.array([0.05, 0.1, 0.15])
        du = np.zeros_like(u)
        t = 0.0
        assert_array_equal(translator.K(u * 0.1, du, t), translator.K(u, du, t))
        assert_equal(np.any(np.not_equal(translator.D(u * 0.1, du, t), translator.D(u, du, t))), True)
        assert_array_equal(translator.M(u * 0.1, du, t), translator.M(u, du, t))
        assert_array_equal(translator.f_ext(u * 0.1, du, t), translator.f_ext(u, du, t))

        assert_array_equal(translator.K(u * 0.1, du, t), K_cust)
        assert_array_equal(translator.M(u * 0.1, du, t), M_cust)
        assert_array_equal(translator.f_ext(u * 0.1, du, t), f_ext_cust)

        self.assertFalse(translator.system_is_linear)
