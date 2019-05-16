"""Test Routine for assembly"""


from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from amfe.constraint.constraint_assembler import ConstraintAssembler


class ConstraintAssemblyTest(TestCase):
    def setUp(self):
        self.casm = ConstraintAssembler()
        self.dofs = [np.array([0, 4]), np.array([1, 5])]
        self.no_of_dofs = 6

        def residual1(x, y):
            return np.array([x[0] + y[0]], dtype=float)

        def residual2(x, y):
            return np.array([x[1] + y[1]], dtype=float)

        self.residuals = [residual1, residual2]
        self.no_of_constraints_by_object = (1, 1)

        def jacobian1(x, y):
            return np.array([x[0], y[0]])

        def jacobian2(x, y):
            return np.array([x[1], y[1]])

        self.jacobians = [jacobian1, jacobian2]

        x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        y = x.copy()

        self.no_of_dofs_unconstrained = 6
        self.args = (x, y)
        self.g_desired = np.array([x[0] + y[0], x[5] + y[5]])
        data = [x[0], y[0], x[5], y[5]]
        row_ind = [0, 0, 1, 1]
        col_ind = [0, 4, 1, 5]
        shape = (2, 6)
        self.B_desired = csr_matrix((data, (row_ind, col_ind)), shape)

    def tearDown(self):
        return

    def test_preallocate_g_and_B(self):
        g_actual, B_actual = self.casm.preallocate_g_and_B(self.no_of_dofs_unconstrained,
                                                           self.dofs, self.no_of_constraints_by_object)

        data_desired = np.zeros(4)
        indices_desired = np.array([0, 4, 1, 5], dtype=int)
        indptr_desired = np.array([0, 2, 4])
        shape_desired = (2, self.no_of_dofs)

        g_desired = np.array([0, 0], dtype=float)

        B_csr_desired = csr_matrix((data_desired, indices_desired, indptr_desired), shape_desired, dtype=float)
        assert_array_equal(B_actual.data, B_csr_desired.data)
        assert_array_equal(B_actual.indptr, B_csr_desired.indptr)
        assert_array_equal(B_actual.indices, B_csr_desired.indices)
        assert_array_equal(g_actual, g_desired)
        return

    def test_assemble_g(self):
        # Test with preallocation
        g, B = self.casm.preallocate_g_and_B(self.no_of_dofs_unconstrained, self.dofs, self.no_of_constraints_by_object)
        g_actual = self.casm.assemble_g(self.residuals, self.dofs, self.args, g)
        assert_array_equal(g_actual, self.g_desired)
        self.assertEqual(id(g), id(g_actual))
        return

    def test_assemble_B(self):
        # Test with preallocation
        g, C = self.casm.preallocate_g_and_B(self.no_of_dofs_unconstrained,
                                             self.dofs, self.no_of_constraints_by_object)
        B_actual = self.casm.assemble_B(self.jacobians, self.dofs, self.args, C)
        assert_array_equal(B_actual.data, self.B_desired.data)
        assert_array_equal(B_actual.indptr, self.B_desired.indptr)
        assert_array_equal(B_actual.indices, self.B_desired.indices)
        self.assertEqual(B_actual.shape, self.B_desired.shape)
        self.assertEqual(id(C), id(B_actual))
        return

    def test_assemble_g_and_B(self):
        # Test with preallocation
        g, B = self.casm.preallocate_g_and_B(self.no_of_dofs_unconstrained,
                                             self.dofs, self.no_of_constraints_by_object)
        B_actual = self.casm.assemble_B(self.jacobians, self.dofs, self.args, B)
        assert_array_equal(B_actual.data, self.B_desired.data)
        assert_array_equal(B_actual.indptr, self.B_desired.indptr)
        assert_array_equal(B_actual.indices, self.B_desired.indices)
        self.assertEqual(B_actual.shape, self.B_desired.shape)
        self.assertEqual(id(B), id(B_actual))
        return

    def test_with_no_constraints(self):
        residuals = []
        jacobians = []
        dofs = []
        no_of_constraints_by_object = ()
        args = []

        # Test preallocation
        g, B = self.casm.preallocate_g_and_B(self.no_of_dofs_unconstrained,
                                             dofs, no_of_constraints_by_object)
        g_desired = np.array([], dtype=float)
        B_desired = csr_matrix((0, self.no_of_dofs))
        assert_array_equal(g, g_desired)
        assert_array_equal(B.data, B_desired.data)
        assert_array_equal(B.indptr, B_desired.indptr)
        assert_array_equal(B.indices, B_desired.indices)

        # Test assembly
        g, B = self.casm.assemble_g_and_B(residuals, jacobians, dofs, args, g, B)
        assert_array_equal(g, g_desired)
        assert_array_equal(B.data, B_desired.data)
        assert_array_equal(B.indptr, B_desired.indptr)
        assert_array_equal(B.indices, B_desired.indices)
