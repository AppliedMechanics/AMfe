'''
Test for testing the translator-module
'''

from unittest import TestCase
import numpy as np
from amfe.solver.translators import *
from numpy.testing import assert_array_equal


class TranslatorsTest(TestCase):
    def setUp(self):
        class DummyStructuralComponent:
            def __init__(self):
                pass
            @property
            def X(self):
                return np.array([0.1, 0.2, 0.3])

            def K(self, q, dq, t):
                return np.array([[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]])

            def M(self, q, dq, t):
                return np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])

            def D(self, q, dq, t):
                return np.array([[0.3, 0, 0], [0, 0.3, 0], [0, 0.3, 0]])

            def f_int(self, q, dq, t):
                return self.K(q, dq, t) @ q

            def K_and_f_int(self, q, dq, t):
                return self.K(q, dq, t), self.f_int(q, dq, t)

            def f_ext(self, q, dq, t):
                return np.array([0., 0., 1])
    
        self.structural_component = DummyStructuralComponent()
        
    def tearDown(self):
        pass
    
    def testMechanicalTranslatorBase(self):
        translator = MechanicalSystemBase(self.structural_component)
        
        u = np.array([0.05, 0.1, 0.15])
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
        assert_array_equal(f_ext_desired - f_int_desired, translator.F(u, du, t))

        u_actual, du_actual, ddu_actual = translator.unconstrain(u, du, du, t)
        for actual, desired in zip((u_actual, du_actual, ddu_actual), (u, du, du)):
            assert_array_equal(actual, desired)
