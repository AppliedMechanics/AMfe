"""Test Routine for component"""


from unittest import TestCase
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

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

