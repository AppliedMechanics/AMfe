# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
io module of AMfe

It handles input output operations for AMfe
"""

import abc

__all__ = [
    'GidAsciiMeshReader',
    'GidBinaryMeshReader',
    'AmfeMeshConverter',
]


class MeshReader(abc.ABC):
    '''
    Abstract super class for all MeshReaders.

    The tasks of the MeshReaders are:
    ---------------------------------
    
    - Read line by line a stream (or file)
    - Call MeshConverter function for each line
    
    '''

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        self._builder = None

    @abc.abstractmethod
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
            raise ValueError('No valid builder given.')


class MeshConverter():
    '''
    Abstract super class for all MeshConverters.
    '''

    def __init__(self, *args, **kwargs):
        pass

    def buildNode(self,x,y,z):
        pass

    def buildElement(self,type,nodes):
        pass

    def buildPhysicalGroup(self,type,id,entities):
        pass

    def buildElementType(self,type):
        pass

    def buildMaterial(self,material):
        pass
