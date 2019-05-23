#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""Test Routine for component-connector"""


from unittest import TestCase
from numpy.testing import assert_array_equal

from amfe.component.component_connector import *


class DummyMesh:
    def __init__(self, dimension):
        """
        Testmesh:                Partition:

                                    9---10--11   11--12
        9---10--11--12              |  *3*  |    |*4*|
        |   |   |   |               |       |    |   |
        |   |   |   |               4---3---6    6---7
        4---3---6---7
        |   |\  |  /|               4---3---6    6---7
        |   |  \| / |               |  *1*  |    |*2*|
        1---2---5---8               |       |    |   |
                                    1---2---5    5---8

        nodes_desired = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0], [3.0, 1.0], [3.0, 0.0], [0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0],
                          [1.0, 1.0], [2.0, 0.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [0.0, 1.0], [3.0, 1.0], [2.0, 2.0]
                          ], dtype=np.float)
        x = nodes_desired[:, 0]
        y = nodes_desired[:, 1]
        nodeids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.nodes_df = pd.DataFrame({'x': x, 'y': y}, index=nodeids)

        connectivity_desired = [np.array([5, 6, 3], dtype=np.int), np.array([3, 2, 5], dtype=np.int),
                        np.array([1, 2, 3, 4], dtype=np.int), np.array([14, 7, 8], dtype=np.int), np.array([15, 7, 14], dtype=np.int),
                        np.array([13, 18, 9, 10], dtype=np.int), np.array([17, 19, 20, 12], dtype=np.int), np.array([13, 16, 10, 11], dtype=np.int),
                        # boundary elements
                        np.array([4, 1], dtype=np.int), np.array([18, 9], dtype=np.int), np.array([7, 8], dtype=np.int), np.array([19, 12], dtype=np.int)]


        data = {'connectivity': connectivity_desired}
        indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        """
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float)
        x = nodes[:, 0]
        y = nodes[:, 1]
        nodeids = [1, 2, 3, 4, 5, 6]
        self.nodes_df = pd.DataFrame({'x': x, 'y': y}, index=nodeids)

        connectivity = [np.array([5, 6, 3], dtype=np.int), np.array([3, 2, 5], dtype=np.int),
                        np.array([1, 2, 3, 4], dtype=np.int),
                        # boundary elements
                        np.array([4, 1], dtype=np.int)]

        data = {'shape': ['Tri3', 'Tri3', 'Quad4', 'straight_line'],
                'is_boundary': [False, False, False, True],
                'connectivity': connectivity,
                'no_of_mesh_partitions': [1, 1, 1, 1],
                'partition_id': [1, 1, 1, 1],
                'partitions_neighbors': [(-2, -3, -4), (-2, -3), -3, -3]}
        indices = [1, 2, 3, 4]
        self._el_df = pd.DataFrame(data, index=indices)

        self.groups = {}

        self.dimension = dimension

    @property
    def no_of_elements(self):
        return 0

    def get_nodeid_by_coordinates(self, x, y, z=None, epsilon=1e-12):
        nodeid = (self.nodes_df[['x', 'y']] - (x, y)).apply(np.linalg.norm, axis=1).idxmin()
        if np.linalg.norm(self.nodes_df.loc[nodeid, ['x', 'y']] - (x, y)) > epsilon:
            nodeid = None
        return nodeid


class DummyMesh2(DummyMesh):
    def __init__(self, dimension):
        super().__init__(dimension)
        """
        Testmesh:                Partition:

                                    9---10--11   11--12
        9---10--11--12              |  *3*  |    |*4*|
        |   |   |   |               |       |    |   |
        |   |   |   |               4---3---6    6---7
        4---3---6---7            
        |   |\  |  /|               4---3---6    6---7
        |   |  \| / |               |  *1*  |    |*2*|
        1---2---5---8               |       |    |   |
                                    1---2---5    5---8
        """
        nodes_desired = np.array([[3.0, 1.0], [3.0, 0.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float)
        x = nodes_desired[:, 0]
        y = nodes_desired[:, 1]
        nodeids = [7, 8, 14, 15]
        self.nodes_df = pd.DataFrame({'x': x, 'y': y}, index=nodeids)

        connectivity = [np.array([14, 7, 8], dtype=np.int), np.array([15, 7, 14], dtype=np.int),
                        # boundary elements
                        np.array([7, 8], dtype=np.int)]

        data = {'shape': ['Tri3', 'Tri3', 'straight_line'],
                'is_boundary': [False, False, True],
                'connectivity': connectivity,
                'no_of_mesh_partitions': [1, 1, 1],
                'partition_id': [2, 2, 2],
                'partitions_neighbors': [(-1, -3, -4), (-1, -4), -4]}
        indices = [1, 2, 3]
        self._el_df = pd.DataFrame(data, index=indices)


class DummyMesh3(DummyMesh):
    def __init__(self, dimension):
        super().__init__(dimension)
        """
        Testmesh:                Partition:

                                    9---10--11   11--12
        9---10--11--12              |  *3*  |    |*4*|
        |   |   |   |               |       |    |   |
        |   |   |   |               4---3---6    6---7
        4---3---6---7            
        |   |\  |  /|               4---3---6    6---7
        |   |  \| / |               |  *1*  |    |*2*|
        1---2---5---8               |       |    |   |
                                    1---2---5    5---8
        """
        nodes_desired = np.array(
            [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
             [1.0, 1.0], [2.0, 1.0], [0.0, 1.0]], dtype=np.float)
        x = nodes_desired[:, 0]
        y = nodes_desired[:, 1]
        nodeids = [9, 10, 11, 13, 16, 18]
        self.nodes_df = pd.DataFrame({'x': x, 'y': y}, index=nodeids)

        connectivity_desired = [np.array([13, 18, 9, 10], dtype=np.int), np.array([13, 16, 10, 11], dtype=np.int),
                                # boundary elements
                                np.array([18, 9], dtype=np.int)]

        data = {'connectivity': connectivity_desired}
        indices = [1, 2, 3]
        self._el_df = pd.DataFrame(data, index=indices)


class DummyMesh5(DummyMesh):
    def __init__(self, dimension):
        super().__init__(dimension)
        """
        Testmesh:                Partition:

                                    9---10--11   11--12
        9---10--11--12              |  *3*  |    |*4*|
        |   |   |   |               |       |    |   |
        |   |   |   |               4---3---6    6---7
        4---3---6---7            
        |   |\  |  /|               4---3---6    6---7
        |   |  \| / |               |  *1*  |    |*2*|
        1---2---5---8               |       |    |   |
                                    1---2---5    5---8
        """
        nodes_desired = np.array([[4.0, 1.0], [4.0, 0.0], [3.0, 0.0], [3.0, 1.0]], dtype=np.float)
        x = nodes_desired[:, 0]
        y = nodes_desired[:, 1]
        nodeids = [7, 8, 14, 15]
        self.nodes_df = pd.DataFrame({'x': x, 'y': y}, index=nodeids)

        connectivity = [np.array([14, 7, 8], dtype=np.int), np.array([15, 7, 14], dtype=np.int),
                        # boundary elements
                        np.array([7, 8], dtype=np.int)]

        data = {'shape': ['Tri3', 'Tri3', 'straight_line'],
                'is_boundary': [False, False, True],
                'connectivity': connectivity,
                'no_of_mesh_partitions': [1, 1, 1],
                'partition_id': [2, 2, 2],
                'partitions_neighbors': [(-1, -3, -4), (-1, -4), -4]}
        indices = [1, 2, 3]
        self._el_df = pd.DataFrame(data, index=indices)


class DummyMeshComponent:
    def __init__(self, mesh):
        self.mesh = mesh
        self._mapping = None

    @property
    def mapping(self):
        return self._mapping

    @property
    def X(self):
        pos = np.array([])
        for node in self.mesh.nodes_df.itertuples():
            pos = np.append(pos, np.array([node.x, node.y]))
        return pos


class ComponentConnectorTest(TestCase):
    def setUp(self):
        class DummyMapping:
            def __init__(self, nodeids):
                dofids_helper = np.arange(2*len(nodeids), dtype=int)
                self.dofids = pd.DataFrame(columns=['nodeid', 'dofs'])
                for i, node in enumerate(nodeids):
                    new_dof = pd.Series({'nodeid': node, 'dofs': dofids_helper[[2*i, 2*i+1]]})
                    self.dofids = self.dofids.append(new_dof, ignore_index=True)
                    self.dofids = self.dofids.astype(dtype={'nodeid': 'int', 'dofs': 'object'})
                self.dofids = self.dofids.set_index('nodeid')

            def get_dofs_by_nodeids(self, nodeids):
                return np.concatenate(self.dofids.loc[nodeids, 'dofs'].values)

            @property
            def no_of_dofs(self):
                return 2 * self.dofids['dofs'].count()

        mesh_master = DummyMesh(2)
        mesh_slave = DummyMesh2(2)
        mesh_nointerf = DummyMesh5(2)
        mesh3 = DummyMesh3(2)

        mapping_master = DummyMapping(mesh_master.nodes_df.index)
        mapping_slave = DummyMapping(mesh_slave.nodes_df.index)
        mapping_nointerf = DummyMapping(mesh_nointerf.nodes_df.index)
        mapping_3 = DummyMapping(mesh3.nodes_df.index)

        self.TestComponent_Master = DummyMeshComponent(mesh_master)
        self.TestComponent_Slave = DummyMeshComponent(mesh_slave)
        self.TestComponent_nointerf = DummyMeshComponent(mesh_nointerf)
        self.TestComponent_3 = DummyMeshComponent(mesh3)

        self.TestComponent_Master._mapping = mapping_master
        self.TestComponent_Slave._mapping = mapping_slave
        self.TestComponent_nointerf._mapping = mapping_nointerf
        self.TestComponent_3._mapping = mapping_3

        self.TestConnector = ComponentConnector()

    def tearDown(self):
        pass

    def test_apply_compatibility_constraint_interf(self):
        C_master_desired = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
                                    dtype=int)
        C_slave_desired = np.array([[0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, -1]],
                                   dtype=int)

        self.TestConnector.apply_compatibility_constraint(1, self.TestComponent_Master, 2, self.TestComponent_Slave)

        assert_array_equal(self.TestConnector.constraints[(2,1)].todense(), C_master_desired)
        assert_array_equal(self.TestConnector.constraints[(1,2)].todense(), C_slave_desired)

        self.TestConnector.apply_compatibility_constraint(2, self.TestComponent_Slave, 1, self.TestComponent_Master)

        assert_array_equal(self.TestConnector.constraints[(2,1)].todense(), -C_master_desired)
        assert_array_equal(self.TestConnector.constraints[(1,2)].todense(), -C_slave_desired)

    def test_apply_compatibility_constraint_nointerf(self):
        self.TestConnector.apply_compatibility_constraint(1, self.TestComponent_Master, 5, self.TestComponent_nointerf)

        self.assertTrue(not self.TestConnector.constraints)

    def test_delete_connection(self):
        self.TestConnector.constraints['testkey'] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

        self.assertTrue('testkey' in self.TestConnector.constraints)

        self.TestConnector.delete_connection('testkey')

        self.assertTrue('testkey' not in self.TestConnector.constraints)

        with self.assertRaises(KeyError): self.TestConnector.delete_connection('wrong_key')

    def test_assemble_constraint_matrices(self):
        self.TestConnector.apply_compatibility_constraint(1, self.TestComponent_Master, 2, self.TestComponent_Slave)
        self.TestConnector.apply_compatibility_constraint(1, self.TestComponent_Master, 3, self.TestComponent_3)
        self.TestConnector.apply_compatibility_constraint(2, self.TestComponent_Slave, 3, self.TestComponent_3)

        B_actual = self.TestConnector._assemble_constraint_matrices(np.array([1, 2, 3, 4], dtype=int),
                                                                    np.array([12, 8, 12, 8], dtype=int))

        def _set_constraint(B, lm_node, nodeidx, value):
            for idx, dofid in enumerate([2 * nodeidx, 2 * nodeidx + 1]):
                B[lm_node * 2 + idx, dofid] = value
            return B

        B_desired_1 = np.zeros((12, 12))
        B_desired_1 = _set_constraint(B_desired_1, 0, 4, 1)
        B_desired_1 = _set_constraint(B_desired_1, 1, 5, 1)
        B_desired_1 = _set_constraint(B_desired_1, 2, 2, 1)
        B_desired_1 = _set_constraint(B_desired_1, 3, 5, 1)
        B_desired_1 = _set_constraint(B_desired_1, 4, 3, 1)

        B_desired_2 = np.zeros((12, 8))
        B_desired_2 = _set_constraint(B_desired_2, 0, 2, -1)
        B_desired_2 = _set_constraint(B_desired_2, 1, 3, -1)
        B_desired_2 = _set_constraint(B_desired_2, 5, 3, 1)

        B_desired_3 = np.zeros((12, 12))
        B_desired_3 = _set_constraint(B_desired_3, 2, 3, -1)
        B_desired_3 = _set_constraint(B_desired_3, 3, 4, -1)
        B_desired_3 = _set_constraint(B_desired_3, 4, 5, -1)
        B_desired_3 = _set_constraint(B_desired_3, 5, 4, -1)

        B_desired = np.hstack((B_desired_1, B_desired_2, B_desired_3, np.zeros((12, 8))))

        assert_array_equal(B_actual, B_desired)


class MeshTyingTest(TestCase):
    def setUp(self):
        self.mesh_master = DummyMesh(2)
        self.mesh_slave = DummyMesh2(2)

    def tearDown(self):
        pass
    
    def test_check_node_compatibility(self):
        nodeids_slave_desired = np.array([14, 15])
        nodeids_master_desired = np.array([5, 6])
        
        mesh_tying = MeshTying()
        nodeids_slave, nodeids_master = mesh_tying.check_node_compatibility(self.mesh_slave, self.mesh_master)
        
        assert_array_equal(nodeids_slave, nodeids_slave_desired)
        assert_array_equal(nodeids_master, nodeids_master_desired)


