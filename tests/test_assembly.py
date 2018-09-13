"""Test Routine for assembly"""


from unittest import TestCase
from scipy.sparse import csr_matrix
from numpy.random import randint
import numpy as np
from scipy import rand
from numpy.testing import assert_array_equal

from amfe.assembly.assembly import Assembly
from amfe.assembly import StructuralAssembly
from amfe.assembly.tools import get_index_of_csr_data, fill_csr_matrix


class AssemblyToolsTest(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_index_of_csr_data(self):
        N = int(10)
        row = randint(0, N, N)
        col = randint(0, N, N)
        val = rand(N)
        A = csr_matrix((val, (row, col)))
        # print('row:', row, '\ncol:', col)
        for i in range(N):
            a = get_index_of_csr_data(row[i], col[i], A.indptr, A.indices)
            b = A[row[i], col[i]]
            #    print(val[i] - A.data[b])
            assert_array_equal(A.data[a], b)

    def test_fill_csr_matrix(self):
        # Test: Build matrix
        #           -                 -     -                 -
        #           | 2.0   0.0   3.0 |     | 1.0   0.0   1.0 |
        #           | 1.0   0.0   1.0 |     | 1.0   0.0   1.0 |
        #           | 4.0   1.0   5.0 |     | 1.0   1.0   1.0 |
        #           -                 -     -                 -

        k_indices = np.array([0, 2])
        K = np.array([[1.0, 2.0], [3.0, 4.0]])
        A_desired = csr_matrix(np.array([[2.0, 0.0, 3.0], [1.0, 0.0, 1.0], [4.0, 1.0, 5.0]], dtype=float))

        dummy = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        A_actual = csr_matrix(dummy)
        fill_csr_matrix(A_actual.indptr, A_actual.indices, A_actual.data, K, k_indices)

        assert_array_equal(A_actual.A, A_desired.A)

    def test_base_class(self):
        class DummyObserver:
            def __init__(self, number):
                self._number = number
                return

            @property
            def number(self):
                return self._number

            @number.setter
            def number(self, no):
                self._number = no
                return

            def update(self, obj):
                self._number = -1
                return

        dummy1 = DummyObserver(1)
        dummy2 = DummyObserver(2)
        assembly = Assembly()
        assembly.add_observer(dummy1)
        assembly.add_observer(dummy2)
        assembly.notify()
        self.assertEqual(dummy1.number, -1)
        self.assertEqual(dummy2.number, -1)

        dummy1.number = 1
        dummy2.number = 1
        assembly.remove_observer(dummy1)
        assembly.notify()
        self.assertEqual(dummy1.number, 1)
        self.assertEqual(dummy2.number, -1)


class StructuralAssemblyTest(TestCase):
    def setUp(self):
        self.nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float)
        self.connectivity = [np.array([4, 5, 2], dtype=np.int), np.array([2, 1, 4], dtype=np.int),
                        np.array([0, 1, 2, 3], dtype=np.int)]
        self.boundary_connectivity = [np.array([3, 0], dtype=np.int), np.array([4, 5], dtype=np.int)]

        self.asm = StructuralAssembly(2, self.nodes, self.connectivity, self.boundary_connectivity)

    def tearDown(self):
        self.asm = None

    def test_mapping(self):
        self.asm.element_mapping = None
        self.asm.compute_element_mapping(self.connectivity, self.boundary_connectivity)

        element_mapping_desired = [np.array([8, 9, 10, 11, 4, 5], dtype=int),
                                   np.array([4, 5, 2, 3, 8, 9], dtype=int),
                                   np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)]
        boundary_element_mapping_desired = [np.array([6, 7, 0, 1], dtype=int), np.array([8, 9, 10, 11], dtype=int)]
        for i, mapping in enumerate(element_mapping_desired):
            assert_array_equal(self.asm.element_mapping[i], element_mapping_desired[i])
        for i, mapping in enumerate(boundary_element_mapping_desired):
            assert_array_equal(self.asm.boundary_element_mapping[i], boundary_element_mapping_desired[i])

    def test_get_dofs_by_nodeidxs(self):
        nodeidxs = np.array([0, 4, 5], dtype=int)
        coords = 'y'
        dofs_actual = self.asm.get_dofs_by_nodeidxs(nodeidxs, coords)
        dofs_desired = np.array([1, 9, 11], dtype=int)
        assert_array_equal(dofs_actual, dofs_desired)
        self.asm.node_mapping[0, 1] = 101
        dofs_actual = self.asm.get_dofs_by_nodeidxs(nodeidxs, coords)
        dofs_desired = np.array([101, 9, 11], dtype=int)
        assert_array_equal(dofs_actual, dofs_desired)

    def test_no_of_dofs_per_node(self):
        actual = self.asm.no_of_dofs_per_node
        self.assertEqual(actual, 2)
