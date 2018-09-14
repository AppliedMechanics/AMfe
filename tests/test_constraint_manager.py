"""Test Routine for constraint manager"""


from unittest import TestCase
import numpy as np
from scipy.sparse import csr_matrix
from numpy.testing import assert_array_equal, assert_raises

from amfe.constraint.structural_constraint_manager import StructuralConstraintManager


class StructuralConstraintManagerTest(TestCase):
    def setUp(self):
        M = np.array([[1,-1,0],[-1,1.2,-1.5],[0,-1.5,2]], dtype=float)
        K = np.array([[2, -1, 0], [-1, 2, -1.5], [0, -1.5, 3]], dtype=float)
        D = 0.2*M + 0.1*K


        self.M_unconstr = csr_matrix(M)
        self.D_unconstr = csr_matrix(D)
        self.K_unconstr = csr_matrix(K)
        self.f_int_unconstr = np.array([1, 2, 3], dtype=float)
        self.f_ext_unconstr = np.array([3, 4, 5], dtype=float)
        self.cm = StructuralConstraintManager(3)

        class DummyDirichletConstraint:
            def __init__(self):
                pass

            def u(self, t):
                return np.array([t])

            def du(self, t):
                return np.array([t])

            def ddu(self, t):
                return np.array([t])

            def slave_dofs(self, dofs_arg):
                return dofs_arg

        self.diric_constraint = DummyDirichletConstraint()

    def tearDown(self):
        self.cm = None

    def testaddconstraint(self):
        self.cm.add_constraint(self.diric_constraint, (2), 'elim')

    def test_constrain_m(self):
        # Constrain third dof
        self.cm.add_constraint(self.diric_constraint, (2), 'elim')
        M_constr = self.cm.constrain_m(self.M_unconstr)
        M_constr_desired = self.M_unconstr[0:2, 0:2]
        assert_array_equal(M_constr.todense(), M_constr_desired.todense())

    def test_constrain_k(self):
        # Constrain third dof
        self.cm.add_constraint(self.diric_constraint, (2), 'elim')
        K_constr = self.cm.constrain_k(self.K_unconstr)
        K_constr_desired = self.K_unconstr[0:2, 0:2]
        assert_array_equal(K_constr.todense(), K_constr_desired.todense())

    def test_constrain_d(self):
        # Constrain third dof
        self.cm.add_constraint(self.diric_constraint, (2), 'elim')
        D_constr = self.cm.constrain_d(self.D_unconstr)
        D_constr_desired = self.D_unconstr[0:2, 0:2]
        assert_array_equal(D_constr.todense(), D_constr_desired.todense())

    def test_constrain_f_int(self):
        self.cm.add_constraint(self.diric_constraint, (2), 'elim')
        f_int_desired = self.f_int_unconstr[0:2]
        f_int = self.cm.constrain_f_int(self.f_int_unconstr)
        assert_array_equal(f_int, f_int_desired)

    def test_constrain_f_ext(self):
        self.cm.add_constraint(self.diric_constraint, (2), 'elim')
        f_ext_desired = self.f_ext_unconstr[0:2]
        f_ext = self.cm.constrain_f_ext(self.f_ext_unconstr)
        assert_array_equal(f_ext, f_ext_desired)

    def test_unconstrain_u(self):
        self.cm.add_constraint(self.diric_constraint, (2), 'elim')
        u_actual = self.cm.unconstrain_u(np.array([0, 0], dtype=float), 3)
        u_desired = np.array([0, 0, 3], dtype=float)
        assert_array_equal(u_actual, u_desired)

    def test_constrain_u(self):
        self.cm.add_constraint(self.diric_constraint, (2), 'elim')
        u_actual = self.cm.constrain_u(np.array([1, 2, 3], dtype=float), 3)
        u_desired = np.array([1, 2], dtype=float)
        assert_array_equal(u_actual, u_desired)

    def test_L(self):
        self.cm.add_constraint(self.diric_constraint, (2), 'elim')
        L_desired = csr_matrix(np.array([[1,0],[0,1],[0,0]], dtype=bool))
        L_actual = self.cm.L
        assert_array_equal(L_actual.todense(), L_desired.todense())

    def test_no_of_constrained_dofs(self):
        self.cm.add_constraint(self.diric_constraint, (2), 'elim')
        no_of_constrained_dofs_actual = self.cm.no_of_constrained_dofs
        no_of_constrained_dofs_desired = 2
        self.assertEqual(no_of_constrained_dofs_actual, no_of_constrained_dofs_desired)

    def test_update_L(self):
        self.cm.add_constraint(self.diric_constraint, (2), 'elim')
        L_old = self.cm.L
        self.cm.add_constraint(self.diric_constraint, (0), 'elim')
        self.cm.update_l()
        L_new = self.cm.L
        assert_raises(AssertionError, assert_array_equal, L_new.todense(), L_old.todense())
        L_desired = L_desired = csr_matrix(np.array([[0],[1],[0]], dtype=bool))
        assert_array_equal(L_new.todense(), L_desired.todense())

    def test_get_rhs_nl(self):
        self.cm.add_constraint(self.diric_constraint, (2), 'elim')
        rhs_actual = self.cm.get_rhs_nl(3, self.M_unconstr, self.D_unconstr)
        rhs_desired = np.array([0,
                                -self.M_unconstr[1,2]*self.diric_constraint.ddu(3)
                                -self.D_unconstr[1,2]*self.diric_constraint.du(3)], dtype=float)
        assert_array_equal(rhs_actual, rhs_desired)

    def test_get_rhs_nl_static(self):
        self.cm.add_constraint(self.diric_constraint, (2), 'elim')
        rhs_actual = self.cm.get_rhs_nl_static(3)
        rhs_desired = np.array([0, 0], dtype=float)
        assert_array_equal(rhs_actual, rhs_desired)

    def test_get_rhs_lin(self):
        self.cm.add_constraint(self.diric_constraint, (2), 'elim')
        rhs_actual = self.cm.get_rhs_lin(3, self.M_unconstr, self.K_unconstr, self.D_unconstr)
        rhs_desired = np.array([0,
                                -self.M_unconstr[1,2]*self.diric_constraint.ddu(3)
                                -self.D_unconstr[1,2]*self.diric_constraint.du(3)
                                -self.K_unconstr[1,2]*self.diric_constraint.u(3)], dtype=float)
        assert_array_equal(rhs_actual, rhs_desired)

    def test_get_rhs_lin_static(self):
        self.cm.add_constraint(self.diric_constraint, (2), 'elim')
        rhs_actual = self.cm.get_rhs_lin_static(3, self.K_unconstr)
        rhs_desired = np.array([0, -self.K_unconstr[1,2]*self.diric_constraint.u(3)], dtype=float)
        assert_array_equal(rhs_actual, rhs_desired)
