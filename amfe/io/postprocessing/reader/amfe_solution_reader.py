#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np
import logging

from ..base import PostProcessorReader
from amfe.io import MeshEntityType, PostProcessDataType


class AmfeSolutionReader(PostProcessorReader):
    def __init__(self, amfesolution, meshcomponent):
        """
        Constructor for AmfeSolutionReader

        Parameters
        ----------
        amfesolution : amfe.solver.solution.AmfeSolution
            Amfe Solution Object
        meshcomponent : amfe.component.MeshComponent
            Mesh Component to which the solution belongs to
        """
        super().__init__()
        self._amfesolution = amfesolution
        self._meshcomponent = meshcomponent
        return

    def parse(self, builder):
        """

        Parameters
        ----------
        builder : amfe.io.postprocessor.PostProcessorWriter

        Returns
        -------

        """
        logger = logging.getLogger(__name__)

        t = np.array(self._amfesolution.t)
        index = self._meshcomponent.mesh.nodes_df.index.values

        if self._amfesolution.q[0] is not None:
            logger.info('Read displacement field from AmfeSolution')
            u_unconstrained = np.array(self._amfesolution.q).T
            displacement_field = self._convert_data_2_field(u_unconstrained)
            builder.write_field('displacement', PostProcessDataType.VECTOR, t, displacement_field, index,
                                MeshEntityType.NODE)

        if self._amfesolution.dq[0] is not None:
            logger.info('Read velocity field from AmfeSolution')
            du_unconstrained = np.array(self._amfesolution.dq).T
            velocity_field = self._convert_data_2_field(du_unconstrained)
            builder.write_field('velocity', PostProcessDataType.VECTOR, t, velocity_field, index,
                                MeshEntityType.NODE)

        if self._amfesolution.ddq[0] is not None:
            logger.info('Read acceleration field from AmfeSolution')
            ddu_unconstrained = np.array(self._amfesolution.ddq).T
            acceleration_field = self._convert_data_2_field(ddu_unconstrained)
            builder.write_field('acceleration', PostProcessDataType.VECTOR, t, acceleration_field, index,
                                MeshEntityType.NODE)

        if self._amfesolution.strain[0] is not None:
            strains = np.array(self._amfesolution.strain).T
            strains_field = self._convert_data_2_normal(strains)
            index = self._meshcomponent.mesh.nodes_df.index.values
            builder.write_field('strains_normal', PostProcessDataType.VECTOR, t, strains_field, index,
                                MeshEntityType.NODE)

        if self._amfesolution.strain[0] is not None:
            strains = np.array(self._amfesolution.strain).T
            strains_field = self._convert_data_2_shear(strains)
            index = self._meshcomponent.mesh.nodes_df.index.values
            builder.write_field('strains_shear', PostProcessDataType.VECTOR, t, strains_field, index,
                                MeshEntityType.NODE)

        if self._amfesolution.stress[0] is not None:
            stresses = np.array(self._amfesolution.stress).T
            stresses_field = self._convert_data_2_normal(stresses)
            index = self._meshcomponent.mesh.nodes_df.index.values
            builder.write_field('stresses_normal', PostProcessDataType.VECTOR, t, stresses_field, index,
                                MeshEntityType.NODE)

        if self._amfesolution.stress[0] is not None:
            stresses = np.array(self._amfesolution.stress).T
            stresses_field = self._convert_data_2_shear(stresses)
            index = self._meshcomponent.mesh.nodes_df.index.values
            builder.write_field('stresses_shear', PostProcessDataType.VECTOR, t, stresses_field, index,
                                MeshEntityType.NODE)

    def _convert_data_2_field(self, data_array):
        """
        Converts a global 1-dimensional array that contains vector field data to a 2-dimensional array.
        The columns of this array represent the x, y and z components of the field.
        The rows contain the data in right order (global node ids)

        Parameters
        ----------
        data_array : ndarray
            1-d-array of vector field data, that shall be converted.

        Returns
        -------
        data : ndarray
            2-d-array, reordered solution-data, columns contain x, y and z components
        """
        data, nodeidxs, nodes2dofs = self._preallocate_dataarray()

        data[nodeidxs, 0, :] = data_array[nodes2dofs[:, 0], :]
        data[nodeidxs, 1, :] = data_array[nodes2dofs[:, 1], :]
        if nodes2dofs.shape[1] > 2:
            data[nodeidxs, 2, :] = data_array[nodes2dofs[:, 2], :]
        else:
            data[nodeidxs, 2, :] = np.zeros((len(nodeidxs), len(self._amfesolution.t)))
        return data

    def _convert_data_2_normal(self, data_array):
        """
        Selects only the first three entries of the data_array, assuming, these are the normal-parts of a strain-/
        stress-tensor and reorders a data-array to match the ordering, which is expected by the Postprocessor-writer.

        Parameters
        ----------
        data_array : ndarray
            array of strains/stresses, that shall be read in.

        Returns
        -------
        data : ndarray
            reordered solution-data
        """
        data, nodeidxs, _ = self._preallocate_dataarray()

        for ientry in range(3):
            data[nodeidxs, ientry, :] = data_array[ientry, nodeidxs, :]
        return data

    def _convert_data_2_shear(self, data_array):
        """
        Selects only the fourth to sixth entries of the data_array, assuming, these are the shear-parts of a strain-/
        stress-tensor and reorders a data-array to match the ordering, which is expected by the Postprocessor-writer.

        Parameters
        ----------
        data_array : ndarray
            array of strains/stresses, that shall be read in.

        Returns
        -------
        data : ndarray
            reordered solution-data
        """
        data, nodeidxs, _ = self._preallocate_dataarray()

        for ientry in range(3, 6):
            data[nodeidxs, ientry-3, :] = data_array[ientry, nodeidxs, :]
        return data

    def _preallocate_dataarray(self):
        no_of_timesteps = len(self._amfesolution.t)

        mapping = self._meshcomponent.mapping
        no_of_nodes = self._meshcomponent.mesh.no_of_nodes
        # Allocate empty array:
        data = np.empty((no_of_nodes, 3, no_of_timesteps))
        data[:, :, :] = np.nan
        nodal2global = mapping.nodal2global.values
        # Write data
        nodeids = mapping._nodal2global.index.values
        nodeidxs = self._meshcomponent.mesh.get_nodeidxs_by_nodeids(nodeids)

        return data, nodeidxs, nodal2global
