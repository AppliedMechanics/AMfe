# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from unittest import TestCase, main
from numpy.testing import assert_equal, assert_allclose
from pandas.testing import assert_frame_equal, assert_series_equal
from amfe.component.partitioner import *
from amfe.component.structural_component import StructuralComponent
from collections.abc import Iterable
from amfe.io.tools import amfe_dir
from amfe.io.mesh import AmfeMeshConverter, GmshAsciiMeshReader
from copy import copy
from .tools import CustomDictAssertTest
import pprint


class DummyMapping:
    def __init__(self, node_ids):
        self.N_dof_types = 3
        self.node_ids = node_ids

    @property
    def no_of_dofs(self):
        return self.N_dof_types * len(self.node_ids)

    def _id2idx(self, id):
        return np.where(self.node_ids == id)[0][0]

    def get_dofs_by_nodeids(self, loc_nodeid, field):
        if field is 'ux':
            return [self.N_dof_types*self._id2idx(loc_nodeid)]
        elif field is 'uy':
            return [self.N_dof_types*self._id2idx(loc_nodeid) + 1]
        elif field is 'uz':
            return [self.N_dof_types*self._id2idx(loc_nodeid) + 2]


class DummyMaterial1:
    def __init__(self):
        pass


class DummyMaterial2:
    def __init__(self):
        pass


class TestPartitioner(TestCase):
    def setUp(self):
        self.custom_asserter = CustomDictAssertTest()
        self.testmesh = Mesh(dimension=2)
        '''
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
        '''
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0], [3.0, 1.0],
                          [3.0, 0.0], [0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0]], dtype=np.float)
        connectivity = [np.array([5, 6, 3], dtype=np.int), np.array([3, 2, 5], dtype=np.int),
                        np.array([1, 2, 3, 4], dtype=np.int), np.array([5, 7, 8], dtype=np.int),
                        np.array([6, 7, 5], dtype=np.int), np.array([3, 4, 9, 10], dtype=np.int),
                        np.array([6, 7, 11, 12], dtype=np.int), np.array([3, 6, 10, 11], dtype=np.int),
                        # boundary elements
                        np.array([4, 1], dtype=np.int), np.array([4, 9], dtype=np.int), np.array([7, 8], dtype=np.int),
                        np.array([7, 12], dtype=np.int)]

        self._connectivity = connectivity

        data = {'shape': ['Tri3', 'Tri3', 'Quad4', 'Tri3', 'Tri3', 'Quad4', 'Quad4', 'Quad4',
                          'straight_line', 'straight_line', 'straight_line', 'straight_line'],
                'is_boundary': [False, False, False, False, False, False, False, False, True, True, True, True],
                'connectivity': connectivity,
                'no_of_mesh_partitions': [4, 3, 2, 3, 4, 2, 4, 4, 2, 2, 2, 2],
                'partition_id': [1, 1, 1, 2, 2, 3, 4, 3, 1, 3, 2, 4],
                'partitions_neighbors': [(2, 3, 4), (2, 3), 3, (1, 4), (1, 3, 4), 1, (1, 2, 3), (1, 2, 4), 3, 1, 4, 2]
                }
        indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        el_df = pd.DataFrame(data, index=indices)

        x = nodes[:, 0]
        y = nodes[:, 1]
        nodeids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        nodes_df = pd.DataFrame({'x': x, 'y': y}, index=nodeids)

        groups = {'left': {'elements': [3], 'nodes': []},
                  'right': {'elements': [1, 2], 'nodes': [2, 3, 5, 6]},
                  'left_boundary': {'elements': [9, 10], 'nodes': [1, 4]},
                  'right_boundary': {'elements': [11, 12], 'nodes': [7, 8]}
                  }

        self.testmesh.nodes_df = nodes_df
        self.testmesh.groups = groups
        self.testmesh._el_df = el_df
        self.no_of_partitions = len(self.testmesh.get_uniques_by_tag('partition_id'))
        
        self.testcomponent = StructuralComponent(self.testmesh)
        self.testcomponent.assign_material(DummyMaterial1(), [1, 2, 5, 6, 7, 8, 9, 10, 11, 12], 'S', '_eleids')
        self.testcomponent.assign_material(DummyMaterial2(), [3, 4], 'S', '_eleids')
        self.partitioner = PartitionedMeshComponentSeparator()

    def tearDown(self):
        pass

    def test_set_partition_tags_by_tag_type(self):
        # Desired nodes
        mesh = Mesh(2)
        nodes = {'x': np.array([0., 0., 5., 5., 10., 10., 0., 10., 2.5, 5., 2.5, 0., 7.17636084, 7.17636084, 5., 2.5, 7.17636084, 5., 0., 2.5, 1.25, 1.25, 3.75, 3.75, 8.22303669, 6.54073581, 6.30235631, 2.84272972, 6.08818042, 8.31613532, 1.33568243]),
                'y': np.array([0., 5., 5., 0., 0., 5., 10., 10., 0., 2.5, 5., 2.5, 0., 5., 2.5, 5., 5., 10., 7.17636084, 2.5, 1.25, 3.75, 1.25, 3.75, 2.55225559, 1.81142646, 3.48052601, 7.44941386, 7.5, 6.875, 6.15644368])}

        mesh.nodes_df = pd.DataFrame.from_dict(nodes)
        mesh.nodes_df.index = range(1, 32)

        # Desired elements
        # (internal name of Triangle Nnode 3 is 'Tri3')
        elements = {'shape': np.array(['straight_line', 'straight_line', 'straight_line', 'straight_line',
                                       'straight_line', 'straight_line', 'Tri3', 'Tri3', 'Tri3', 'Tri3',
                                       'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3',
                                       'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3',
                                       'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3',
                                       'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3',
                                       'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3'], dtype=object),
                    'is_boundary': np.array([ True,  True,  True,  True,  True,  True, False, False, False,
                                           False, False, False, False, False, False, False, False, False,
                                           False, False, False, False, False, False, False, False, False,
                                           False, False, False, False, False, False, False, False, False,
                                           False, False, False, False, False, False, False, False, False,
                                           False, False, False]),
                    'connectivity': np.array([np.array([ 2, 12]), np.array([12,  1]), np.array([5, 6]), np.array([6, 8]),
                                           np.array([ 7, 19]), np.array([19,  2]), np.array([24, 23, 10]),
                                           np.array([20, 23, 24]), np.array([ 4, 23,  9]), np.array([ 3, 24, 10]),
                                           np.array([ 9, 23, 20]), np.array([ 9, 20, 21]), np.array([ 1, 21, 12]),
                                           np.array([12, 21, 20]), np.array([11, 20, 24]), np.array([11, 22, 20]),
                                           np.array([12, 20, 22]), np.array([ 3, 11, 24]), np.array([ 2, 22, 11]),
                                           np.array([ 2, 12, 22]), np.array([ 4, 10, 23]), np.array([ 1,  9, 21]),
                                           np.array([ 5, 25, 13]), np.array([ 6, 14, 25]), np.array([14, 27, 25]),
                                           np.array([13, 25, 26]), np.array([ 4, 13, 26]), np.array([ 4, 26, 15]),
                                           np.array([25, 27, 26]), np.array([ 3, 15, 27]), np.array([ 3, 27, 14]),
                                           np.array([15, 26, 27]), np.array([ 5,  6, 25]), np.array([ 7, 28, 18]),
                                           np.array([ 3, 28, 16]), np.array([ 7, 19, 28]), np.array([ 3, 29, 28]),
                                           np.array([ 8, 29, 30]), np.array([ 3, 17, 29]), np.array([ 8, 18, 29]),
                                           np.array([19, 31, 28]), np.array([ 2, 16, 31]), np.array([16, 28, 31]),
                                           np.array([ 6, 30, 17]), np.array([ 2, 31, 19]), np.array([17, 30, 29]),
                                           np.array([18, 28, 29]), np.array([ 6,  8, 30])], dtype=object),
                    'no_of_mesh_partitions': np.array([1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    'partition_id': np.array([2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
                    'partitions_neighbors': np.array([None, None, 2, 1, None, None, None, 2, None, None, 2, 2, 1, 1, 2, 1, 1, 2, 1, None, None, 2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                                                      None, None, None, None, None, None, None, None, None], dtype=object),
                    'elemental_group': np.array([4,  4,  6, 11, 13, 13,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3, 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3])}

        mesh._el_df = pd.DataFrame.from_dict(elements)
        mesh._el_df.index = range(1, 49)

        groups = {'surface_left': {'elements': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                                    'nodes': []},
                 'surface_right': {'elements': [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                                   'nodes': []},
                 'surface_top': {'elements': [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
                                 'nodes': []},
                 'x_dirichlet-line': {'elements': [1, 2, 5, 6], 'nodes': []},
                 'x_neumann': {'elements': [3, 4], 'nodes': []}}

        mesh.groups = groups

        #Desired Mesh
        mesh_desired = Mesh(2)
        mesh_desired.nodes_df = copy(mesh.nodes_df)

        elements_desired = {'shape': np.array(['straight_line', 'straight_line', 'straight_line', 'straight_line',
                                       'straight_line', 'straight_line', 'Tri3', 'Tri3', 'Tri3', 'Tri3',
                                       'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3',
                                       'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3',
                                       'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3',
                                       'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3',
                                       'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3', 'Tri3'], dtype=object),
                    'is_boundary': np.array([True, True, True, True, True, True, False, False, False,
                                             False, False, False, False, False, False, False, False, False,
                                             False, False, False, False, False, False, False, False, False,
                                             False, False, False, False, False, False, False, False, False,
                                             False, False, False, False, False, False, False, False, False,
                                             False, False, False]),
                    'connectivity': np.array([np.array([2, 12]), np.array([12, 1]), np.array([5, 6]), np.array([6, 8]),
                                              np.array([7, 19]), np.array([19, 2]), np.array([24, 23, 10]),
                                              np.array([20, 23, 24]), np.array([4, 23, 9]), np.array([3, 24, 10]),
                                              np.array([9, 23, 20]), np.array([9, 20, 21]), np.array([1, 21, 12]),
                                              np.array([12, 21, 20]), np.array([11, 20, 24]), np.array([11, 22, 20]),
                                              np.array([12, 20, 22]), np.array([3, 11, 24]), np.array([2, 22, 11]),
                                              np.array([2, 12, 22]), np.array([4, 10, 23]), np.array([1, 9, 21]),
                                              np.array([5, 25, 13]), np.array([6, 14, 25]), np.array([14, 27, 25]),
                                              np.array([13, 25, 26]), np.array([4, 13, 26]), np.array([4, 26, 15]),
                                              np.array([25, 27, 26]), np.array([3, 15, 27]), np.array([3, 27, 14]),
                                              np.array([15, 26, 27]), np.array([5, 6, 25]), np.array([7, 28, 18]),
                                              np.array([3, 28, 16]), np.array([7, 19, 28]), np.array([3, 29, 28]),
                                              np.array([8, 29, 30]), np.array([3, 17, 29]), np.array([8, 18, 29]),
                                              np.array([19, 31, 28]), np.array([2, 16, 31]), np.array([16, 28, 31]),
                                              np.array([6, 30, 17]), np.array([2, 31, 19]), np.array([17, 30, 29]),
                                              np.array([18, 28, 29]), np.array([6, 8, 30])], dtype=object),
                    'no_of_mesh_partitions': np.array(
                        [2, 2, 2, 2, 1, 2, 2, 2, 2, 3, 2, 2 ,2, 2, 3, 3, 2, 4, 3, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 3, 3, 2,
                         2, 1, 4, 1, 3, 1, 3, 1, 1, 3, 3, 2, 2, 2, 1, 2]),
                    'partition_id': np.array(
                        [1, 1, 3, 4, 4, 4, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                         3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]),
                    'partitions_neighbors': np.array(
                        [4, 2, 4, 3, None, 1, 3, 1, 3, (3, 4), 1, 1, 2, 2, (1, 4), (2, 4), 2, (1, 3, 4), (2, 4), 4, 3, 1,
                         None, 4, 4, None, 2, 2, None, (2, 4), (2, 4), 2, 4, None, (1, 2, 3), None, (2, 3), None,
                         (2, 3), None, None, (1, 2), (1, 2), 3, 1, 3, None, 3], dtype=object),
                    'elemental_group': np.array(
                        [4, 4, 6, 11, 13, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                         2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])}

        mesh_desired._el_df = pd.DataFrame.from_dict(elements_desired)
        mesh_desired._el_df.index = range(1, 49)

        groups_desired = {'surface_left': {'elements': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                                   'nodes': []},
                  'surface_right': {'elements': [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                                    'nodes': []},
                  'surface_top': {'elements': [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
                                  'nodes': []},
                  'x_dirichlet-line': {'elements': [1, 2, 5, 6], 'nodes': []},
                  'x_neumann': {'elements': [3, 4], 'nodes': []}}

        mesh_desired.groups = groups_desired

        self.partitioner.set_partition_tags_by_group(mesh, 'surface')

        self.custom_asserter.assert_dict_equal(mesh._el_df['shape'], mesh_desired._el_df['shape'])
        self.custom_asserter.assert_dict_equal(mesh._el_df['is_boundary'], mesh_desired._el_df['is_boundary'])
        self.custom_asserter.assert_dict_equal(mesh._el_df['connectivity'], mesh_desired._el_df['connectivity'])
        self.custom_asserter.assert_dict_equal(mesh._el_df['partition_id'], mesh_desired._el_df['partition_id'])
        self.custom_asserter.assert_dict_equal(mesh._el_df['partitions_neighbors'],
                                               mesh_desired._el_df['partitions_neighbors'])
        self.custom_asserter.assert_dict_equal(mesh._el_df['no_of_mesh_partitions'], mesh_desired._el_df['no_of_mesh_partitions'])
        self.custom_asserter.assert_dict_equal(mesh._el_df['elemental_group'], mesh_desired._el_df['elemental_group'])

    def test_separate_common_nodes_of_partitions(self):
        nodes_desired = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0], [3.0, 1.0],
                                  [3.0, 0.0], [0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0],
                                  [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [0.0, 1.0], [2.0, 1.0], [2.0, 1.0], [3.0, 1.0],
                                  [2.0, 2.0]
                                  ], dtype=np.float)
        x = nodes_desired[:, 0]
        y = nodes_desired[:, 1]
        nodeids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        nodes_df_desired = pd.DataFrame({'x': x, 'y': y}, index=nodeids)
        
        connectivity_desired = [np.array([5, 6, 3], dtype=np.int), np.array([3, 2, 5], dtype=np.int),
                                np.array([1, 2, 3, 4], dtype=np.int), np.array([13, 7, 8], dtype=np.int),
                                np.array([14, 7, 13], dtype=np.int),
                                np.array([15, 16, 9, 10], dtype=np.int), np.array([18, 19, 20, 12], dtype=np.int),
                                np.array([15, 17, 10, 11], dtype=np.int),
                                # boundary elements
                                np.array([4, 1], dtype=np.int), np.array([16, 9], dtype=np.int),
                                np.array([7, 8], dtype=np.int), np.array([19, 12], dtype=np.int)]

        data = {'connectivity': connectivity_desired}
        indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        el_df_desired = pd.DataFrame(data, index=indices)
        
        groups_desired = {'left': {'elements': [3], 'nodes': []},
                          'right': {'elements': [1, 2], 'nodes': [2, 3, 5, 6, 13, 14, 15, 17, 18]},
                          'left_boundary': {'elements': [9, 10], 'nodes': [1, 4, 16]},
                          'right_boundary': {'elements': [11, 12], 'nodes': [7, 8, 19]}
                          }

        new_mesh, nodes_map_actual = self.partitioner._separate_common_nodes_of_partitions(self.testmesh)

        nodes_map_desired = pd.DataFrame({'partition_id': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
                                          'global_nodeid': [1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 3, 4, 6, 9, 10, 11, 6, 7, 11,
                                                            12],
                                          'local_nodeid': [1, 2, 3, 4, 5, 6, 13, 14, 7, 8, 15, 16, 17, 9, 10, 11, 18,
                                                           19, 20, 12]})

        assert_allclose(new_mesh.nodes_df, nodes_df_desired)
        assert_series_equal(new_mesh._el_df['connectivity'], el_df_desired['connectivity'])
        assert_equal(new_mesh.groups, groups_desired)
        assert_frame_equal(nodes_map_actual, nodes_map_desired)

    def test_get_submesh_by_partition_id(self):
        separated_mesh, copied_nodes = self.partitioner._separate_common_nodes_of_partitions(self.testmesh)
        submesh = self.partitioner._get_submesh_by_partition_id(1, separated_mesh)
        
        nodes_desired = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
                                 dtype=np.float)
        x = nodes_desired[:, 0]
        y = nodes_desired[:, 1]
        nodeids = [1, 2, 3, 4, 5, 6]
        nodes_df_desired = pd.DataFrame({'x': x, 'y': y}, index=nodeids)
        
        connectivity_desired = [np.array([5, 6, 3], dtype=np.int), np.array([3, 2, 5], dtype=np.int),
                                np.array([1, 2, 3, 4], dtype=np.int), np.array([4, 1], dtype=np.int)]
        data = {'connectivity': connectivity_desired}
        indices = [1, 2, 3, 9]
        el_df_desired = pd.DataFrame(data, index=indices)
        
        groups_desired = {'left': {'elements': [3], 'nodes': []},
                  'right': {'elements': [1, 2], 'nodes': [2, 3, 5, 6]},
                  'left_boundary': {'elements': [9], 'nodes': [1, 4]}
                  }
        
        self.assertTrue(isinstance(submesh, Mesh))
        assert_allclose(submesh.nodes_df, nodes_df_desired)
        assert_series_equal(submesh._el_df['connectivity'], el_df_desired['connectivity'])
        assert_equal(submesh.groups, groups_desired)

        submesh = self.partitioner._get_submesh_by_partition_id(4, separated_mesh)
        
        nodes_desired = np.array([[3.0, 2.0], [2.0, 1.0], [3.0, 1.0], [2.0, 2.0]], dtype=np.float)
        x = nodes_desired[:, 0]
        y = nodes_desired[:, 1]
        nodeids = [12, 16, 19, 20]
        nodes_df_desired = pd.DataFrame({'x': x, 'y': y}, index=nodeids)
        
        connectivity_desired = [np.array([18, 19, 20, 12], dtype=np.int), np.array([19, 12], dtype=np.int)]
        data = {'connectivity': connectivity_desired}
        indices = [7, 12]
        el_df_desired = pd.DataFrame(data, index=indices)
        
        groups_desired = {'right': {'elements': [], 'nodes': [18]},
                          'right_boundary': {'elements': [12], 'nodes': [19]}
                          }
        
        self.assertTrue(isinstance(submesh, Mesh))
        assert_allclose(submesh.nodes_df, nodes_df_desired)
        assert_series_equal(submesh._el_df['connectivity'], el_df_desired['connectivity'])
        assert_equal(submesh.groups, groups_desired)

    def test_map_dofs_local2global(self):
        d = {'partition_id': [3, 3, 3, 3, 3, 3],
             'global_nodeid': [3, 4, 6, 9, 10, 11],
             'local_nodeid': [15, 16, 17, 9, 10, 11]}
        nodes_mapping_df = pd.DataFrame(data=d)

        mesh, nodes_mapping = self.partitioner._separate_common_nodes_of_partitions(self.testcomponent._mesh)
        local_component = StructuralComponent(self.partitioner._get_submesh_by_partition_id(3, mesh))

        local_component.assign_material(DummyMaterial1(), [6, 8, 10], 'S', '_eleids')

        dofs_map_actual = self.partitioner._map_dofs_local2global(nodes_mapping_df, local_component._mapping,
                                                                  self.testcomponent._mapping,
                                                                  self.testcomponent.fields)

        dofs_map_desired = {0: 4, 1: 5, 2: 10, 3: 11, 8: 2, 9: 3, 4: 12, 5: 13, 6: 14, 7: 15, 10: 16, 11: 17}

        assert_equal(dofs_map_desired, dofs_map_actual)

    def test_separate_partitioned_component(self):
        new_component_ids, new_components, dof_map_list = self.partitioner.separate_partitioned_component(
                                                                                                    self.testcomponent)
        component_ids_desired = [1, 2, 3, 4]
        
        for comp_id, icomp in zip(new_component_ids, new_components):
            self.assertTrue(isinstance(icomp, StructuralComponent))
            self.assertTrue(comp_id in component_ids_desired)
            assert_equal(len(icomp.mesh.get_uniques_by_tag('partition_id')), 1)
            for idx, element in icomp._ele_obj_df.iterrows():
                if not isinstance(element['fk_mesh'], Iterable):
                    eleids = [element['fk_mesh']]
                else:
                    eleids = element['fk_mesh']
                if 3 in eleids or 4 in eleids:
                    self.assertTrue(isinstance(element['ele_obj'].material, DummyMaterial2))
                else:
                    self.assertTrue(isinstance(element['ele_obj'].material, DummyMaterial1))

        assert_equal(len(new_components), self.no_of_partitions)
        assert_equal(len(dof_map_list), self.no_of_partitions)


if __name__ == '__main__':
    main()
