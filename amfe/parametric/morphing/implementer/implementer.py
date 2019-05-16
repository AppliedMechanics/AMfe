#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import abc

__all__ = ['MorpherImplementer'
           ]


class MorpherImplementer:
    """
    Implements the morphing technique (e.g. FFD or RBF)
    """
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def offline(self, nodes_reference):
        """
        This method can be called once to initialize the morpher and let it calculate variables that must be calculated
        once

        Parameters
        ----------
        nodes_reference : ndarray
            ndarray with node coordinates (rows = nodes, columns = x,y,z coordinate)

        Returns
        -------
        None
        """
        pass

    @abc.abstractmethod
    def morph(self, *args):
        """

        Parameters
        ----------
        args[0] : ndarray
            node coordinates
        args[1:] : unknown
            morpher specific parameters

        Returns
        -------
        morphed nodes : ndarray
            node coordinates of the morphed nodes
        """
        return args[0]
