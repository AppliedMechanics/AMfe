"""Test Routine for assembly"""


from unittest import TestCase
from scipy.sparse import csr_matrix
from numpy.random import randint
import numpy as np
from scipy import rand
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.testing import assert_frame_equal

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
        #           --               --     --               --
        #           | 2.0   0.0   3.0 |     | 1.0   0.0   1.0 |
        #           | 1.0   0.0   1.0 |     | 1.0   0.0   1.0 |
        #           | 4.0   1.0   5.0 |     | 1.0   1.0   1.0 |
        #           --               --     --               --

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
        self.nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float)
        self.iconnectivity = [np.array([0, 1, 2], dtype=np.int), np.array([0, 2, 3], dtype=np.int),
                              np.array([1, 2], dtype=np.int), np.array([2, 3], dtype=np.int)]

        self.asm = StructuralAssembly()

        class DummyTri3Element:
            def __init__(self):
                pass

            def m_int(self, X, u, t=0.):
                M = np.array([[2, 0, -0.5, 0, -0.5, 0],
                              [0, 2, 0, -0.5, 0, -0.5],
                              [-0.5, 0, 2, 0, -0.5, 0],
                              [0, -0.5, 0, 2, 0, -0.5],
                              [-0.5, 0, -0.5, 0, 2, 0],
                              [0, -0.5, 0, -0.5, 0, 2]], dtype=float)
                return M

            def k_and_f_int(self, X, u, t=0.):
                K = np.array([[4, -0.5, -0.5, -0.2, -0.5, -0.2],
                              [-0.2, 4, -0.2, -0.5, -0.2, -0.5],
                              [-0.5, -0.2, 4, -0.2, -0.5, -0.2],
                              [-0.2, -0.5, -0.2, 4, -0.2, -0.5],
                              [-0.5, -0.2, -0.5, -0.2, 4, -0.2],
                              [-0.2, -0.5, -0.2, -0.5, -0.2, 4]], dtype=float)
                f = np.array([3, 1, 3, 1, 3, 1], dtype=float)
                return K, f

            def k_f_S_E_int(self, X, u, t=0):
                K, f = self.k_and_f_int(X, u, t)
                S = np.ones((3, 6), dtype=float)
                E = 2*np.ones((3, 6), dtype=float)
                return K, f, S, E

        self.ele = DummyTri3Element()

    def tearDown(self):
        self.asm = None

    def test_preallocate_csr(self):
        no_of_dofs = 6
        elements2global = np.array([np.array([0, 1, 2, 3], dtype=int), np.array([2, 3, 4, 5], dtype=int)])
        C_csr_actual = self.asm.preallocate(no_of_dofs, elements2global)

        vals = np.zeros(32)
        rows = [0, 1, 2, 3]*4
        rows.extend([2, 3, 4, 5]*4)
        cols = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        cols.extend([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])
        C_csr_desired = csr_matrix((vals, (rows, cols)), dtype=float)
        assert_array_equal(C_csr_actual.data, C_csr_desired.data)
        assert_array_equal(C_csr_actual.indptr, C_csr_desired.indptr)
        assert_array_equal(C_csr_actual.indices, C_csr_desired.indices)

    def test_assemble_m(self):

        asm = StructuralAssembly()

        ele_obj = np.array([self.ele, self.ele], dtype=object)
        elements2global = np.array([np.array([0, 1, 2, 3, 4, 5], dtype=int), np.array([0, 1, 4, 5, 6, 7], dtype=int)])

        M_global = asm.preallocate(8, elements2global)

        memory_m_before = id(M_global)
        asm.assemble_m(self.nodes, ele_obj, self.iconnectivity[0:2], elements2global, M_csr=M_global)
        memory_m_after = id(M_global)
        # test if preallocated version works
        self.assertTrue(memory_m_after, memory_m_before)

        M_global_desired = np.zeros((8, 8), dtype=float)
        # element 1
        M_global_desired[0:6, 0:6] = self.ele.m_int(None, None)

        # element 2
        # diagonals
        M_global_desired[0:2, 0:2] += self.ele.m_int(None, None)[0:2, 0:2]
        M_global_desired[4:, 4:] += self.ele.m_int(None, None)[2:, 2:]
        # off-diagonals
        M_global_desired[0:2, 4:] += self.ele.m_int(None, None)[0:2, 2:]
        M_global_desired[4:, 0:2] += self.ele.m_int(None, None)[2:, 0:2]

        assert_array_equal(M_global.todense(), M_global_desired)

    def test_assemble_k_and_f(self):
        asm = StructuralAssembly()
        ele_obj = np.array([self.ele, self.ele], dtype=object)
        element2dofs = np.array([np.array([0, 1, 2, 3, 4, 5], dtype=int), np.array([0, 1, 4, 5, 6, 7], dtype=int)])
        K_global = asm.preallocate(8, element2dofs)
        f_global = np.zeros(K_global.shape[0])

        asm.assemble_k_and_f(self.nodes, ele_obj, self.iconnectivity[0:2], element2dofs, K_csr=K_global, f_glob=f_global)
        K_global_desired = np.zeros((8, 8), dtype=float)
        f_global_desired = np.zeros(8, dtype=float)
        # element 1
        K_global_desired[0:6, 0:6], f_global_desired[0:6] = self.ele.k_and_f_int(None, None)

        # element 2
        K_local, f_local = self.ele.k_and_f_int(None, None)
        # diagonals
        K_global_desired[0:2, 0:2] += K_local[0:2, 0:2]
        K_global_desired[4:, 4:] += K_local[2:, 2:]
        # off-diagonals
        K_global_desired[0:2, 4:] += K_local[0:2, 2:]
        K_global_desired[4:, 0:2] += K_local[2:, 0:2]
        # f_int:
        f_global_desired[0:2] += f_local[0:2]
        f_global_desired[4:] += f_local[2:]

        assert_array_equal(K_global.todense(), K_global_desired)
        assert_array_equal(f_global, f_global_desired)

    def test_assemble_k_and_f_preallocation(self):

        asm = StructuralAssembly()
        ele_obj = np.array([self.ele, self.ele], dtype=object)
        element2dofs = np.array([np.array([0, 1, 2, 3, 4, 5], dtype=int), np.array([0, 1, 4, 5, 6, 7], dtype=int)])
        K_global = asm.preallocate(8, element2dofs)
        f_global = np.zeros(K_global.shape[0])
        memory_K_global_before = id(K_global)
        memory_K_global_data_before = id(K_global.data)
        memory_f_global_before = id(f_global)

        K_global, f_global = asm.assemble_k_and_f(self.nodes, ele_obj, self.iconnectivity[0:2], element2dofs, K_csr=K_global)

        memory_K_global_after = id(K_global)
        memory_K_global_data_after = id(K_global.data)
        memory_f_global_after = id(f_global)

        self.assertTrue(memory_K_global_after == memory_K_global_before)
        self.assertTrue(memory_K_global_data_after == memory_K_global_data_before)
        self.assertFalse(memory_f_global_after == memory_f_global_before)

        memory_K_global_before = memory_K_global_after
        memory_K_global_data_before = memory_K_global_data_after
        memory_f_global_before = memory_f_global_after

        K_global, f_global = asm.assemble_k_and_f(self.nodes, ele_obj, self.iconnectivity[0:2], element2dofs, f_glob=f_global)

        memory_K_global_after = id(K_global)
        memory_K_global_data_after = id(K_global.data)
        memory_f_global_after = id(f_global)

        self.assertFalse(memory_K_global_after == memory_K_global_before)
        self.assertFalse(memory_K_global_data_after == memory_K_global_data_before)
        self.assertTrue(memory_f_global_after == memory_f_global_before)

    def test_assemble_k_f_S_E(self):
        asm = StructuralAssembly()
        ele_obj = np.array([self.ele, self.ele], dtype=object)
        element2dofs = [np.array([0, 1, 2, 3, 4, 5], dtype=int), np.array([0, 1, 4, 5, 6, 7], dtype=int)]
        elements_on_node = np.array([2, 1, 2, 1], dtype=float)
        C_csr = asm.preallocate(8, element2dofs)
        f_global = np.zeros(C_csr.shape[0])

        memory_K_global_before = id(C_csr)
        memory_K_global_data_before = id(C_csr.data)
        memory_f_global_before = id(f_global)

        C_csr, f_global, S_global, E_global= asm.assemble_k_f_S_E(self.nodes, ele_obj, self.iconnectivity[0:2],
                                                                     element2dofs, elements_on_node, K_csr=C_csr, f_glob=f_global, )

        memory_K_global_after = id(C_csr)
        memory_K_global_data_after = id(C_csr.data)
        memory_f_global_after = id(f_global)

        # test fully preallocated version
        self.assertTrue(memory_K_global_after == memory_K_global_before)
        self.assertTrue(memory_K_global_data_after == memory_K_global_data_before)
        self.assertTrue(memory_f_global_after == memory_f_global_before)

        K_global_desired = np.zeros((8, 8), dtype=float)
        f_global_desired = np.zeros(8, dtype=float)
        # Currently strains and stresses are always 3D (therefore 6 components)
        S_global_desired = np.ones((4, 6))
        E_global_desired = np.ones((4, 6))*2

        # element 1
        K_global_desired[0:6, 0:6], f_global_desired[0:6], _, _ = self.ele.k_f_S_E_int(None, None)

        # element 2
        K_local, f_local, S_local, E_local = self.ele.k_f_S_E_int(None, None)
        # diagonals
        K_global_desired[0:2, 0:2] += K_local[0:2,0:2]
        K_global_desired[4:, 4:] += K_local[2:, 2:]
        # off-diagonals
        K_global_desired[0:2, 4:] += K_local[0:2, 2:]
        K_global_desired[4:, 0:2] += K_local[2:, 0:2]
        # f_int:
        f_global_desired[0:2] += f_local[0:2]
        f_global_desired[4:] += f_local[2:]

        assert_array_equal(C_csr.todense(), K_global_desired)
        assert_array_equal(f_global, f_global_desired)
        assert_array_equal(S_global, S_global_desired)
        assert_array_equal(E_global, E_global_desired)


