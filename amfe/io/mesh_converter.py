#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Super class of all mesh converter for I/O module.
"""

__all__ = [
    'MeshConverter'
    ]


class MeshConverter:
    """
    Super class for all mesh converters.
    """

    def __init__(self, *args, **kwargs):
        pass

    def build_no_of_nodes(self, no):
        pass

    def build_no_of_elements(self, no):
        pass

    def build_node(self, id, x, y, z):
        pass

    def build_element(self, id, type, nodes):
        pass

    def build_group(self, name, nodeids, elementids):
        """

        Parameters
        ----------
        name: str
            Name identifying the node group.
        nodeids: list
            List with node ids.
        elementids: list
            List with element ids.

        Returns
        -------
        None
        """

        pass

    def build_material(self, material):
        pass

    def build_partition(self, partition):
        pass

    def build_mesh_dimension(self, dim):
        pass

    def build_tag(self, tag_dict):
        """
        Builds a tag with following dict given in tag_dict

        Parameters
        ----------
        tag_dict : dict
            dict with following format:
            { tagname1 : { tagvalue1 : [elementids],
                           tagvalue2 : [elementids],
                           ...
                         },
              tagname2 : { tagvalue1 : [elementids],
                           tagvalue2 : [elementids]
                           ...
                         },
              ...
            }

        Returns
        -------
        None
        """
        pass

    def return_mesh(self):
        pass
