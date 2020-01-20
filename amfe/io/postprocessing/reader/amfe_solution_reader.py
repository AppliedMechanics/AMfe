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
        data[nodeidxs, 0, :] = data_array[nodal2global[:, 0], :]
        data[nodeidxs, 1, :] = data_array[nodal2global[:, 1], :]
        if nodal2global.shape[1] > 2:
            data[nodeidxs, 2, :] = data_array[nodal2global[:, 2], :]
        else:
            data[nodeidxs, 2, :] = np.zeros((len(nodeidxs), no_of_timesteps))
        return data
