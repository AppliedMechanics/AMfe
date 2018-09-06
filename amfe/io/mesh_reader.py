#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Abstract super class of all mesh reader for I/O module.
"""

from abc import ABC, abstractmethod

from .mesh_converter import MeshConverter

__all__ = [
    'MeshReader'
    ]


class MeshReader(ABC):
    """
    Abstract super class for all mesh readers.

    TASKS
    -----
    - Read line by line a stream (or file).
    - Call mesh converter functions for each line.

    NOTES
    -----
    PLEASE FOLLOW THE BUILDER PATTERN!
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self._builder = None
        return

    @abstractmethod
    def parse(self):
        pass

    @property
    def builder(self):
        return self._builder

    @builder.setter
    def builder(self, builder):
        if isinstance(builder, MeshConverter):
            self._builder = builder
        else:
            raise ValueError('Invalid builder given.')
        return
