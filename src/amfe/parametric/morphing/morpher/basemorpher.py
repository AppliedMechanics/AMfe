#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import abc


__all__ = ['MeshMorpher']


class MeshMorpher:
    """
    MeshMorpher base class for all mesh morphing techniques

    Provides a common interface
    """
    KWARGS = ()

    # set basic properties
    def __init__(self, *args, **kwargs):
        pass

    # Two phases: 1. Initialization = offline(), 2. Morphing: = morph()
    @abc.abstractmethod
    def offline(self, nodes_reference):
        pass

    @abc.abstractmethod
    def morph(self, **kwargs):
        """

        Parameters
        ----------
        kwargs: key value pairs can differ

        Returns
        -------
        nodes
        """
        pass
