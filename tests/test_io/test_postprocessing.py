"""
Tests for testing postprocessing-module of io
"""

from unittest import TestCase
import numpy as np
import os
import h5py
from numpy.testing import assert_array_equal
from copy import copy
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


def create_fields(dim=3):
    amfemesh = create_amfe_obj()
    meshreader = AmfeMeshObjMeshReader(amfemesh)

    timesteps = np.arange(0, 0.8, 0.2)  # 4 timesteps
    no_of_nodes = amfemesh.no_of_nodes
    no_of_cells = amfemesh.no_of_elements
    no_of_dofs = no_of_nodes * dim
    # q = np.random.rand(no_of_dofs * len(timesteps)).reshape(no_of_dofs, len(timesteps))
    q = np.ones((no_of_dofs, len(timesteps)))
    q[:, 0] = q[:, 0] * 0.0
    q[:, 1] = q[:, 1] * 0.1
    q[:, 2] = q[:, 2] * 0.2
    q[:, 3] = q[:, 3] * 0.3
    q2 = -q
    strains = np.ones((6, no_of_nodes, len(timesteps)))
    factor = 0.0
    strains[:, :, 0] *= factor
    factor = 0.1
    for t in range(1, len(timesteps)):
        for dir in range(0, 6):
            factor += 0.02
            strains[dir, :, t] *= factor
    stresses = strains * 1e3

    s = np.arange(no_of_cells * len(timesteps)).reshape(no_of_cells, len(timesteps))
    volume_indices = amfemesh.el_df[amfemesh.el_df['is_boundary'] == False].index.values

    fields_desired = {'Nodefield1': {'data_type': PostProcessDataType.VECTOR, 'timesteps': timesteps,
                                          'data': q, 'index': amfemesh.nodes_df.index.values,
                                          'mesh_entity_type': MeshEntityType.NODE},
                           'Nodefield2': {'data_type': PostProcessDataType.VECTOR, 'timesteps': timesteps,
                                          'data': q2, 'index': amfemesh.nodes_df.index.values,
                                          'mesh_entity_type': MeshEntityType.NODE},
                           'NodefieldStrains': {'data_type': PostProcessDataType.VECTOR, 'timesteps': timesteps,
                                          'data': strains, 'index': amfemesh.nodes_df.index.values,
                                          'mesh_entity_type': MeshEntityType.NODE},
                           'NodefieldStresses': {'data_type': PostProcessDataType.VECTOR, 'timesteps': timesteps,
                                           'data': stresses, 'index': amfemesh.nodes_df.index.values,
                                           'mesh_entity_type': MeshEntityType.NODE},
                           'Elementfield1': {'data_type': PostProcessDataType.SCALAR, 'timesteps': timesteps,
                                             'data': s, 'index': volume_indices,
                                             'mesh_entity_type': MeshEntityType.ELEMENT}
                           }
    fields_no_of_nodes = no_of_nodes
    fields_no_of_timesteps = len(timesteps)

    return timesteps, meshreader, fields_desired, fields_no_of_nodes, fields_no_of_timesteps


class PostProcessorTest(TestCase):
    def setUp(self):
        clean_test_outputs()

    def tearDown(self):
        pass

    def test_hdf5_postprocessor_writer_and_reader(self):
        self.timesteps, self.meshreader, self.fields_desired, \
        self.fields_no_of_nodes, self.fields_no_of_timesteps = create_fields()

        filename = amfe_dir('results/.tests/hdf5postprocessing.hdf5')

        if os.path.isfile(filename):
            os.remove(filename)

        writer = Hdf5PostProcessorWriter(self.meshreader, filename, '/myresults')
        fields_desired = {'Nodefield1': self.fields_desired['Nodefield1'],
                          'Nodefield2': self.fields_desired['Nodefield2'],
                          'Elementfield1': self.fields_desired['Elementfield1']}
        fields = copy(fields_desired)
        for fieldname in fields:
            field = fields[fieldname]
            if field['data_type'] == PostProcessDataType.VECTOR:
                data = field['data'].reshape(self.fields_no_of_nodes, 3, self.fields_no_of_timesteps)
            else:
                data = field['data']
            writer.write_field(fieldname, field['data_type'], field['timesteps'],
                               data, field['index'], field['mesh_entity_type'])

        self.timesteps, self.meshreader, self.fields_desired, \
        self.fields_no_of_nodes, self.fields_no_of_timesteps = create_fields()

        h5filename = amfe_dir('results/.tests/hdf5postprocessing.hdf5')

        postprocessorreader = AmfeHdf5PostProcessorReader(h5filename,
                                                          meshrootpath='/mesh',
                                                          resultsrootpath='/myresults')
        meshreader = Hdf5MeshReader(h5filename, '/mesh')
        postprocessorwriter = DummyPostProcessorWriter(meshreader)
        postprocessorreader.parse(postprocessorwriter)
        fields = postprocessorwriter.return_result()
        # Check no of fields:
        self.assertEqual(len(fields.keys()), len(fields_desired.keys()))
        # Check each field:
        for fieldname in fields_desired:
            field_actual = fields[fieldname]
            field_desired = fields_desired[fieldname]
            assert_array_equal(field_actual['timesteps'], field_desired['timesteps'])
            assert_array_equal(field_actual['data_type'], field_desired['data_type'])
            assert_array_equal(field_actual['data'], field_desired['data'])
            assert_array_equal(field_actual['index'], field_desired['index'])
            assert_array_equal(field_actual['mesh_entity_type'], field_desired['mesh_entity_type'])

    def test_write_xdmf_from_hdf5(self):
        self.timesteps, self.meshreader, self.fields_desired, \
        self.fields_no_of_nodes, self.fields_no_of_timesteps = create_fields()
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


class AmfeSolutionReaderTest(TestCase):
    def setUp(self):
        self.timesteps, self.meshreader, self.fields_desired, \
        self.fields_no_of_nodes, self.fields_no_of_timesteps = create_fields(2)

        amfesolution = AmfeSolution()
        sol = self.fields_desired['Nodefield1']
        strains = copy(self.fields_desired['NodefieldStrains'])
        stresses = copy(self.fields_desired['NodefieldStresses'])
        for i in range(len(sol['timesteps'])):
            t = sol['timesteps'][i]
            q = sol['data'][:, i].T
            strain = strains['data'][:, :, i].T
            stress = stresses['data'][:, :, i].T
            amfesolution.write_timestep(t, q, q, q, strain, stress)

        self.mesh = create_amfe_obj()
        self.meshcomponent = StructuralComponent(self.mesh)
        # Set a material to get a mapping
        material = KirchhoffMaterial()
        self.meshcomponent.assign_material(material, 'Tri6', 'S', 'shape')

        self.postprocessorreader = AmfeSolutionReader(amfesolution, self.meshcomponent)

    def tearDown(self):
        pass

    def test_parse(self):
        #meshreader = AmfeMeshObjMeshReader(self.mesh)
        postprocessorwriter = DummyPostProcessorWriter(self.meshreader)
        self.postprocessorreader.parse(postprocessorwriter)
        fields_actual = postprocessorwriter.return_result()

        field_desired = self.fields_desired['Nodefield1']
        q = field_desired['data']
        dofs_x = self.meshcomponent.mapping.get_dofs_by_nodeids(self.meshcomponent.mesh.nodes_df.index.values, ('ux'))
        dofs_y = self.meshcomponent.mapping.get_dofs_by_nodeids(self.meshcomponent.mesh.nodes_df.index.values, ('uy'))
        q_x = q[dofs_x, :]
        q_y = q[dofs_y, :]
        data = np.empty((0, 3, 4), dtype=float)
        for node in self.meshcomponent.mesh.get_nodeidxs_by_all():
            data = np.concatenate((data, np.array([[q_x[node], q_y[node], np.zeros(q_x.shape[1])]])), axis=0)
        field_desired['data'] = data

        def _get_desired_strain_stress_fields(nodeidxs, field):
            normal_desired = copy(field)
            shear_desired = copy(field)
            values = field['data']
            data1 = np.empty((0, 3, 4), dtype=float)
            data2 = np.empty((0, 3, 4), dtype=float)

            for node in nodeidxs:
                data1 = np.concatenate((data1, np.array([[values[0, node], values[1, node],
                                                          values[2, node]]])), axis=0)
                data2 = np.concatenate((data2, np.array([[values[3, node], values[4, node],
                                                          values[5, node]]])), axis=0)

            normal_desired['data'] = data1
            shear_desired['data'] = data2
            return normal_desired, shear_desired

        strains_normal_desired, strains_shear_desired = \
            _get_desired_strain_stress_fields(self.meshcomponent.mesh.get_nodeidxs_by_all(),
                                              self.fields_desired['NodefieldStrains'])

        stresses_normal_desired, stresses_shear_desired = \
            _get_desired_strain_stress_fields(self.meshcomponent.mesh.get_nodeidxs_by_all(),
                                              self.fields_desired['NodefieldStresses'])
        # Check no of fields:
        self.assertEqual(len(fields_actual.keys()), 7)
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
        field_acceleration_actual = fields_actual['strains_normal']
        assert_array_equal(field_acceleration_actual['timesteps'], strains_normal_desired['timesteps'])
        assert_array_equal(field_acceleration_actual['data_type'], strains_normal_desired['data_type'])
        assert_array_equal(field_acceleration_actual['data'], strains_normal_desired['data'])
        assert_array_equal(field_acceleration_actual['index'], strains_normal_desired['index'])
        assert_array_equal(field_acceleration_actual['mesh_entity_type'], strains_normal_desired['mesh_entity_type'])
        field_acceleration_actual = fields_actual['strains_shear']
        assert_array_equal(field_acceleration_actual['timesteps'], strains_shear_desired['timesteps'])
        assert_array_equal(field_acceleration_actual['data_type'], strains_shear_desired['data_type'])
        assert_array_equal(field_acceleration_actual['data'], strains_shear_desired['data'])
        assert_array_equal(field_acceleration_actual['index'], strains_shear_desired['index'])
        assert_array_equal(field_acceleration_actual['mesh_entity_type'], strains_shear_desired['mesh_entity_type'])
        field_acceleration_actual = fields_actual['stresses_normal']
        assert_array_equal(field_acceleration_actual['timesteps'], stresses_normal_desired['timesteps'])
        assert_array_equal(field_acceleration_actual['data_type'], stresses_normal_desired['data_type'])
        assert_array_equal(field_acceleration_actual['data'], stresses_normal_desired['data'])
        assert_array_equal(field_acceleration_actual['index'], stresses_normal_desired['index'])
        assert_array_equal(field_acceleration_actual['mesh_entity_type'], stresses_normal_desired['mesh_entity_type'])
        field_acceleration_actual = fields_actual['stresses_shear']
        assert_array_equal(field_acceleration_actual['timesteps'], stresses_shear_desired['timesteps'])
        assert_array_equal(field_acceleration_actual['data_type'], stresses_shear_desired['data_type'])
        assert_array_equal(field_acceleration_actual['data'], stresses_shear_desired['data'])
        assert_array_equal(field_acceleration_actual['index'], stresses_shear_desired['index'])
        assert_array_equal(field_acceleration_actual['mesh_entity_type'], stresses_shear_desired['mesh_entity_type'])

    def test_convert_data_2_field(self):
        sol = self.fields_desired['Nodefield1']
        data_test = sol['data']
        field_actual = self.postprocessorreader._convert_data_2_field(data_test)

        dofs_x = self.meshcomponent.mapping.get_dofs_by_nodeids(self.meshcomponent.mesh.nodes_df.index.values, ('ux'))
        dofs_y = self.meshcomponent.mapping.get_dofs_by_nodeids(self.meshcomponent.mesh.nodes_df.index.values, ('uy'))
        q_x = data_test[dofs_x, :]
        q_y = data_test[dofs_y, :]
        field_desired = np.empty((0, 3, 4), dtype=float)
        for node in self.meshcomponent.mesh.get_nodeidxs_by_all():
            field_desired = np.concatenate((field_desired,
                                            np.array([[q_x[node], q_y[node], np.zeros(q_x.shape[1])]])), axis=0)

        assert_array_equal(field_actual, field_desired)

    def test_convert_data_2_normal(self):
        sol = self.fields_desired['NodefieldStrains']
        data_test = sol['data']
        field_actual = self.postprocessorreader._convert_data_2_normal(data_test)

        field_desired = np.empty((0, 3, 4), dtype=float)
        for node in self.meshcomponent.mesh.get_nodeidxs_by_all():
            field_desired = np.concatenate((field_desired, np.array([[data_test[0, node], data_test[1, node],
                                                      data_test[2, node]]])), axis=0)

        assert_array_equal(field_actual, field_desired)

    def test_convert_data_2_shear(self):
        sol = self.fields_desired['NodefieldStrains']
        data_test = sol['data']
        field_actual = self.postprocessorreader._convert_data_2_shear(data_test)

        field_desired = np.empty((0, 3, 4), dtype=float)
        for node in self.meshcomponent.mesh.get_nodeidxs_by_all():
            field_desired = np.concatenate((field_desired, np.array([[data_test[3, node], data_test[4, node],
                                                      data_test[5, node]]])), axis=0)

        assert_array_equal(field_actual, field_desired)



