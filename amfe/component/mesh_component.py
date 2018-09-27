#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

from copy import deepcopy
import numpy as np

from amfe.mesh import Mesh
from .component_base import ComponentBase
from amfe.component.constants import ELEPROTOTYPEHELPERLIST

__all__ = ['MeshComponent']


class MeshComponent(ComponentBase):

    # The following class attributes must be overwritten by subclasses
    ELEMENTPROTOTYPES = dict(((element[0], None) for element in ELEPROTOTYPEHELPERLIST))

    def __init__(self, mesh=Mesh()):
        super().__init__()
        self._mesh = mesh
        self._ele_obj = np.empty(mesh.no_of_elements, dtype=object)

    # -- PROPERTIES --------------------------------------------------------------------------------------
    @property
    def ele_obj(self):
        return self._ele_obj

    # -- ASSIGN MATERIAL METHODS -------------------------------------------------------------------------
    def assign_material(self, materialobj, propertynames, tag='_groups'):
        if tag == '_groups':
            eleidxes = self._mesh.get_elementidxs_by_groups(propertynames)
        elif tag == '_eleidxs':
            eleidxes = propertynames
        else:
            eleidxes = self._mesh.get_elementidxs_by_tags(propertynames)
        self._assign_material_by_eleidxs(materialobj, eleidxes)

    def _assign_material_by_eleidxs(self, materialobj, eleidxes):
        prototypes = deepcopy(self.ELEMENTPROTOTYPES)
        for prototype in prototypes.values():
            prototype.material = materialobj
        ele_shapes = self._mesh.get_ele_shapes_by_idxs(eleidxes)
        self._ele_obj[eleidxes] = [prototypes[ele_shape] for ele_shape in ele_shapes]

    # -- ASSIGN NEUMANN CONDITION METHODS -----------------------------------------------------------------

    # -- ASSIGN CONSTRAINTS METHODS ------------------------------------------------------------------------

    # -- MAPPING METHODS -----------------------------------------------------------------------------------

    # -- GETTER FOR SYSTEM MATRICES ------------------------------------------------------------------------
    #
    # MUST BE IMPLEMENTED IN SUBCLASSES
    #
