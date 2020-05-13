"""
Tests for testing postprocessing-module of io
"""

from unittest import TestCase
import numpy as np
import os
import h5py
from numpy.testing import assert_array_equal
# Import Postprocessing Tools
from amfe.io import AmfeMeshObjMeshReader, AmfeSolutionReader, Hdf5PostProcessorWriter, AmfeHdf5PostProcessorReader, \
    Hdf5MeshReader, write_xdmf_from_hdf5
from amfe.io.postprocessing.base import PostProcessorWriter
from amfe.io.constants import PostProcessDataType, MeshEntityType
from .tools import create_amfe_obj, clean_test_outputs
from amfe.io.tools import amfe_dir
from amfe.solver import AmfeSolution
from amfe.component import StructuralComponent
from amfe.material import KirchhoffMaterial


class DummyPostProcessorWriter(PostProcessorWriter):
    def __init__(self, meshreaderobj):
        super().__init__(meshreaderobj)
        self._meshreader = meshreaderobj
        self._fields = dict()

    def write_field(self, name, field_type, t, data, index, mesh_entity_type):
        fielddict = {'data_type': field_type,
                     'timesteps': t,
                     'index': index,
                     'mesh_entity_type': mesh_entity_type,
                     'data': data
                     }
        if name in fielddict:
            raise ValueError('Field already written')
        self._fields.update({name: fielddict})

    def return_result(self):
        return self._fields


class PostProcessorTest(TestCase):
    def setUp(self):
        clean_test_outputs()

    def tearDown(self):
        pass

    def _create_fields(self, dim=3):
        amfemesh = create_amfe_obj()
        self.meshreader = AmfeMeshObjMeshReader(amfemesh)

        self.timesteps = np.arange(0, 0.8, 0.2)  # 4 timesteps
        no_of_nodes = amfemesh.no_of_nodes
        no_of_cells = amfemesh.no_of_elements
        no_of_dofs = no_of_nodes * dim
        # q = np.random.rand(no_of_dofs * len(timesteps)).reshape(no_of_dofs, len(timesteps))
        q = np.ones((no_of_dofs, len(self.timesteps)))
        q[:, 0] = q[:, 0] * 0.0
        q[:, 1] = q[:, 1] * 0.1
        q[:, 2] = q[:, 2] * 0.2
        q[:, 3] = q[:, 3] * 0.3
        q2 = -q

        s = np.arange(no_of_cells * len(self.timesteps)).reshape(no_of_cells, len(self.timesteps))
        volume_indices = amfemesh.el_df[amfemesh.el_df['is_boundary'] == False].index.values

        self.fields_desired = {'Nodefield1': {'data_type': PostProcessDataType.VECTOR, 'timesteps': self.timesteps,
                                              'data': q, 'index': amfemesh.nodes_df.index.values,
                                              'mesh_entity_type': MeshEntityType.NODE},
                               'Nodefield2': {'data_type': PostProcessDataType.VECTOR, 'timesteps': self.timesteps,
                                              'data': q2, 'index': amfemesh.nodes_df.index.values,
                                              'mesh_entity_type': MeshEntityType.NODE},
                               'Elementfield1': {'data_type': PostProcessDataType.SCALAR, 'timesteps': self.timesteps,
                                                 'data': s, 'index': volume_indices,
                                                 'mesh_entity_type': MeshEntityType.ELEMENT}
                               }
        self.fields_no_of_nodes = no_of_nodes
        self.fields_no_of_timesteps = len(self.timesteps)

    def test_hdf5_postprocessor_writer_and_reader(self):
        self._create_fields()

        filename = amfe_dir('results/.tests/hdf5postprocessing.hdf5')

        if os.path.isfile(filename):
            os.remove(filename)

        writer = Hdf5PostProcessorWriter(self.meshreader, filename, '/myresults')
        fields = self.fields_desired
        for fieldname in fields:
            field = fields[fieldname]
            if field['data_type'] == PostProcessDataType.VECTOR:
                data = field['data'].reshape(self.fields_no_of_nodes, 3, self.fields_no_of_timesteps)
            else:
                data = field['data']
            writer.write_field(fieldname, field['data_type'], field['timesteps'],
                               data, field['index'], field['mesh_entity_type'])

        self._create_fields()

        h5filename = amfe_dir('results/.tests/hdf5postprocessing.hdf5')

        postprocessorreader = AmfeHdf5PostProcessorReader(h5filename,
                                                          meshrootpath='/mesh',
                                                          resultsrootpath='/myresults')
        meshreader = Hdf5MeshReader(h5filename, '/mesh')
        postprocessorwriter = DummyPostProcessorWriter(meshreader)
        postprocessorreader.parse(postprocessorwriter)
        fields = postprocessorwriter.return_result()
        # Check no of fields:
        self.assertEqual(len(fields.keys()), len(self.fields_desired.keys()))
        # Check each field:
        for fieldname in self.fields_desired:
            field_actual = fields[fieldname]
            field_desired = self.fields_desired[fieldname]
            assert_array_equal(field_actual['timesteps'], field_desired['timesteps'])
            assert_array_equal(field_actual['data_type'], field_desired['data_type'])
            assert_array_equal(field_actual['data'], field_desired['data'])
            assert_array_equal(field_actual['index'], field_desired['index'])
            assert_array_equal(field_actual['mesh_entity_type'], field_desired['mesh_entity_type'])

    def test_write_xdmf_from_hdf5(self):
        self._create_fields()
        filename = amfe_dir('results/.tests/hdf5postprocessing.hdf5')
        with h5py.File(filename, mode='r') as hdf5_fp:
            filename = amfe_dir('results/.tests/hdf5postprocessing.xdmf')
            with open(filename, 'wb') as xdmf_fp:
                fielddict = self.fields_desired
                for key in fielddict:
                    fielddict[key].update({'hdf5path': '/myresults/{}'.format(key)})
                    timesteps = fielddict[key]['timesteps']
                # timesteps = np.arange(0, 0.8, 0.2)  # 4 timesteps
                write_xdmf_from_hdf5(xdmf_fp, hdf5_fp, '/mesh/nodes', '/mesh/topology', timesteps, fielddict)

    def test_amfe_solution_reader(self):
        self._create_fields(2)

        amfesolution = AmfeSolution()
        sol = self.fields_desired['Nodefield1']
        for t, q in zip(sol['timesteps'], sol['data'].T):
            amfesolution.write_timestep(t, q, q, q)

        mesh = create_amfe_obj()
        meshcomponent = StructuralComponent(mesh)
        # Set a material to get a mapping
        material = KirchhoffMaterial()
        meshcomponent.assign_material(material, 'Tri6', 'S', 'shape')

        postprocessorreader = AmfeSolutionReader(amfesolution, meshcomponent)

        meshreader = AmfeMeshObjMeshReader(mesh)
        postprocessorwriter = DummyPostProcessorWriter(meshreader)
        postprocessorreader.parse(postprocessorwriter)
        fields_actual = postprocessorwriter.return_result()

        field_desired = sol
        q = field_desired['data']
        dofs_x = meshcomponent.mapping.get_dofs_by_nodeids(meshcomponent.mesh.nodes_df.index.values, ('ux'))
        dofs_y = meshcomponent.mapping.get_dofs_by_nodeids(meshcomponent.mesh.nodes_df.index.values, ('uy'))
        q_x = q[dofs_x, :]
        q_y = q[dofs_y, :]
        data = np.empty((0, 3, 4), dtype=float)
        for node in meshcomponent.mesh.get_nodeidxs_by_all():
                data = np.concatenate((data, np.array([[q_x[node], q_y[node], np.zeros(q_x.shape[1])]])), axis=0)
        field_desired['data'] = data
        # Check no of fields:
        self.assertEqual(len(fields_actual.keys()), 3)
        # Check each field:
        field_displacement_actual = fields_actual['displacement']
        assert_array_equal(field_displacement_actual['timesteps'], field_desired['timesteps'])
        assert_array_equal(field_displacement_actual['data_type'], field_desired['data_type'])
        assert_array_equal(field_displacement_actual['data'], field_desired['data'])
        assert_array_equal(field_displacement_actual['index'], field_desired['index'])
        assert_array_equal(field_displacement_actual['mesh_entity_type'], field_desired['mesh_entity_type'])
        field_velocity_actual = fields_actual['velocity']
        assert_array_equal(field_velocity_actual['timesteps'], field_desired['timesteps'])
        assert_array_equal(field_velocity_actual['data_type'], field_desired['data_type'])
        assert_array_equal(field_velocity_actual['data'], field_desired['data'])
        assert_array_equal(field_velocity_actual['index'], field_desired['index'])
        assert_array_equal(field_velocity_actual['mesh_entity_type'], field_desired['mesh_entity_type'])
        field_acceleration_actual = fields_actual['acceleration']
        assert_array_equal(field_acceleration_actual['timesteps'], field_desired['timesteps'])
        assert_array_equal(field_acceleration_actual['data_type'], field_desired['data_type'])
        assert_array_equal(field_acceleration_actual['data'], field_desired['data'])
        assert_array_equal(field_acceleration_actual['index'], field_desired['index'])
        assert_array_equal(field_acceleration_actual['mesh_entity_type'], field_desired['mesh_entity_type'])