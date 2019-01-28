import numpy as np

from ..base import PostProcessorReader
from .. import MeshEntityType, PostProcessDataType


class AmfeSolutionReader(PostProcessorReader):
    def __init__(self, amfesolution, meshcomponent, is_constrained=True):
        """
        Constructor for AmfeSolutionReader

        Parameters
        ----------
        amfesolution : amfe.solver.solution.AmfeSolution
            Amfe Solution Object
        meshcomponent : amfe.component.MeshComponent
            Mesh Component to which the solution belongs to
        is_constrained : bool
            flag if the solution is a constrained dofs solution or unconstrained
        """
        super().__init__()
        self._amfesolution = amfesolution
        self._meshcomponent = meshcomponent
        self._is_constrained = is_constrained
        return

    def parse(self, builder):
        """

        Parameters
        ----------
        builder : amfe.io.postprocessor.PostProcessorWriter

        Returns
        -------

        """
        if self._is_constrained:
        # Get full vector: u_full = f(q).... not implemented yet
            raise NotImplementedError('is_constrained option not implemented yet')
        else:
            raise NotImplementedError('This function is not implemented yet. It can be implemented'
                                      'when the solvers are finished in refactoring')

            t = np.array(self._amfesolution.t)
            u = np.array(self._amfesolution.q)
            if self._amfesolution.dq[0] is not None or self._amfesolution.ddq[0] is not None:
                raise NotImplementedError('Velocities and Accelerations cannot be written')
            mapping = self._meshcomponent._mapping.nodal2global.values.reshape(-1)
            for i, u_current in enumerate(u):
                u[i, :] = u_current[mapping]
            index = self._meshcomponent._mapping.nodal2global.index
            builder.write_field('displacement', PostProcessDataType.VECTOR, t, u, index, MeshEntityType.NODE)
