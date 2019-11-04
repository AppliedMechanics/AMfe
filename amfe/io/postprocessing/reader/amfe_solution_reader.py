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
        no_of_timesteps = len(self._amfesolution.t)
        t = np.array(self._amfesolution.t)
        u_unconstrained = np.array(self._amfesolution.q).T
        if self._amfesolution.dq[0] is not None or self._amfesolution.ddq[0] is not None:
            logger = logging.getLogger(__name__)
            logger.warning('Velocities and Accelerations cannot be written by AmfeSolutionReader')
        mapping = self._meshcomponent.mapping
        no_of_nodes = self._meshcomponent.mesh.no_of_nodes
        # Allocate empty array:
        data = np.empty((no_of_nodes, 3, no_of_timesteps))
        data[:, :, :] = np.nan
        nodal2global = mapping.nodal2global.values
        # Write ux
        nodeids = mapping._nodal2global.index.values
        nodeidxs = self._meshcomponent.mesh.get_nodeidxs_by_nodeids(nodeids)
        data[nodeidxs, 0, :] = u_unconstrained[nodal2global[:, 0], :]
        data[nodeidxs, 1, :] = u_unconstrained[nodal2global[:, 1], :]
        if nodal2global.shape[1] > 2:
            data[nodeidxs, 2, :] = u_unconstrained[nodal2global[:, 2], :]
        else:
            data[nodeidxs, 2, :] = np.zeros((len(nodeidxs), no_of_timesteps))

        index = self._meshcomponent.mesh.nodes_df.index.values
        builder.write_field('displacement', PostProcessDataType.VECTOR, t, data, index, MeshEntityType.NODE)
