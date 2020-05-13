import numpy as np
import pandas as pd
from tables import File as PytablesFile, open_file

from amfe.io.postprocessing.base import PostProcessorWriter
from .. import MeshEntityType, PostProcessDataType
from ..amfe_postprocess_mesh_converter import AmfePostprocessMeshConverter
from amfe.io.tools import check_filename_or_filepointer
from amfe.io.mesh.writer import Hdf5MeshConverter

__all__ = ['Hdf5PostProcessorWriter']


class Hdf5PostProcessorWriter(PostProcessorWriter):
    def __init__(self, meshreaderobj, filename, rootpath='/results', verbose_rootname=None):
        """
        Constructor for the Hdf5PostProcessorWriter

        Parameters
        ----------
        meshreader : amfe.io.mesh_reader.MeshReader
            MeshReader object that can parse the mesh information of the Postprocessor Data
        filename : str or table.File
            HDF5-filename or tables.File object where the converted data shall be stored to
        rootpath : str
            str describing the HDF5-path where the results shall be written to (default: '/results')
        verbose_rootname : str
            A verbose rootname can be given if desired (default: None)
        """
        super().__init__(meshreaderobj)
        self._filename = filename
        self._rootpath = rootpath
        self._verbose_rootname = verbose_rootname
        if verbose_rootname is None:
            self._verbose_rootname = 'No name'
        self._fp = None
        # write mesh at first
        self._mesh_container = None
        self._write_mesh()
        self._written_fields = set()

    def write_field(self, name, field_type, t, data, index, mesh_entity_type):
        """
        Parameters
        ----------
        name : str
            Name for the field to write
        field_type : PostProcessDataType
            Data Type
        t : ndarray
            Timesteps
        data : ndarray
            ndarray to write
        index : ndarray
            ndarray with indices of the mesh entities in the mesh reader object
        mesh_entity_type : MeshEntityType
            type of mesh entities

        Returns
        -------
        None
        """
        return self._write_field(self._filename, name, field_type, t, data, index, mesh_entity_type)

    def return_result(self):
        """

        Returns
        -------
        filename : str or tables.File
        """
        return self._filename

    @check_filename_or_filepointer(PytablesFile, open_file, 1, writeable=True)
    def _write_field(self, fp, name, field_type, t, data, index, mesh_entity_type):
        """
        Parameters
        ----------
        fp : tables.File
            tables.File object to write field into
        name : str
            Name for the field to write
        field_type : PostProcessDataType
            Data Type
        t : ndarray
            Timesteps
        data : ndarray
            ndarray to write
        index : ndarray
            ndarray with indices of the mesh entities in the mesh reader object
        mesh_entity_type : MeshEntityType
            type of mesh entities

        Returns
        -------
        None
        """
        if name in self._written_fields:
            raise ValueError('Field {} already written. Cannot write the same field again'.format(name))

        # Create root path for results if necessary
        if self._rootpath not in fp:
            resultsroot = fp.create_group('/', self._rootpath[1:], self._verbose_rootname)
        else:
            resultsroot = fp.get_node(self._rootpath)

        # Create timesteps if not written before, otherwise check if it is consistent with current timesteps
        if 'timesteps' not in resultsroot:
            timesteps = np.array(t).astype(float)
            fp.create_array(resultsroot, 'timesteps', timesteps, 'Timesteps', shape=timesteps.shape)
        else:
            if not np.array_equal(fp.get_node(resultsroot, 'timesteps', classname='Array').read(), np.array(t)):
                raise ValueError('The timesteps of the current field is not compatible with the timesteps of'
                                 'already written field within this group')

        # -- PROCEDURE FOR ELEMENT DATA --
        if mesh_entity_type == MeshEntityType.ELEMENT:
            # Element data must be stored within a folder containing separate arrays for each shape
            # Thus create this folder:
            fieldgroup = fp.create_group(resultsroot, name)

            # Get the Element Dataframe from the mesh
            el_df = self._mesh_container['elements']

            # Get an array with unique element shapes
            etypes = el_df['shape'].unique()

            # Write an ascending enumeration at those places given by the index argument into the dataframe
            el_df['isource'] = None
            el_df.loc[index, 'isource'] = np.arange(len(index))
            for etype in etypes:
                # For each etype write an ascending enumeration into 'igoal' column.
                # This enumeration is seperate for each etype
                el_df_current_etype = el_df[el_df['shape'] == etype].copy()
                no_of_elements_of_current_etype = len(el_df_current_etype.index)
                el_df_current_etype.loc[el_df_current_etype.index, 'igoal'] = np.arange(no_of_elements_of_current_etype)

                # allocate arr_to_write with default value nan
                arr_to_write = np.zeros((no_of_elements_of_current_etype, len(t)))
                arr_to_write[:] = np.nan

                # get all elements of current etype where data is available (given by index argument,
                # this is why whe have set el_df['isource'] = None by default and then overwrote them at loc[index]
                # with an ascending enumeration)
                el_df_current_etype_not_nan = el_df_current_etype.loc[pd.notna(el_df_current_etype['isource']), ['igoal','isource']]
                isource = el_df_current_etype_not_nan['isource'].values.astype(int)
                igoal = el_df_current_etype_not_nan['igoal'].values.astype(int)

                # Write the data of data array at the right places
                if field_type == PostProcessDataType.SCALAR and data.ndim == 1:
                    data = data.reshape(-1, 1)
                arr_to_write[igoal, :] = data[isource, :]

                # write the array for current etype into the hdf5 and its attribute information
                dataset = fp.create_array(fieldgroup, etype, arr_to_write, name, shape=arr_to_write.shape)
                dataset.attrs.data_type = field_type.name
                dataset.attrs.mesh_entitiy_type = mesh_entity_type.name

        # -- PROCEDURE FOR NODE DATA --
        else:
            # Get the ilocs of the given index for nodes
            nodesiloc = self._mesh_container['nodes'].loc[index, 'iloc'].values.astype(int)
            # if vector than change the nodesiloc such because vector has 3 following elements in data array
            if field_type == PostProcessDataType.VECTOR:
                data[nodesiloc, :, :] = data[:, :, :]
                data = data.reshape(len(nodesiloc)*3, len(t))
            elif field_type == PostProcessDataType.SCALAR:
                data[nodesiloc, :] = data[:, :]
            else:
                raise NotImplementedError('Field Data Type {} is not supported for converting'.format(field_type.name))

            dataset = fp.create_array(resultsroot, name, data)
            dataset.attrs.data_type = field_type.name
            dataset.attrs.mesh_entitiy_type = mesh_entity_type.name
        self._written_fields.add(name)

    def _write_mesh(self):
        mesh_converter = Hdf5MeshConverter(self._filename)
        self._meshreader.parse(mesh_converter)
        mesh_converter.return_mesh()
        mesh_converter = AmfePostprocessMeshConverter()
        self._meshreader.parse(mesh_converter)
        self._mesh_container = mesh_converter.return_mesh()
