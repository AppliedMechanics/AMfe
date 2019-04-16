# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

"""
This module contains constants for the postprocessors
"""

from enum import Enum

__all__ = ['PostProcessDataType',
           'MeshEntityType',
           'XDMFDICT',
           'MESHENTITYTYPE2XDMF',
           'POSTPROCESSDATATYPE2XDMF',
           'POSTPROCESSDATATYPE2XDMFDIMENSION'
           ]


class PostProcessDataType(Enum):
    """
    PostProcessDataType is an Enum describing different Data Types for postprocessordata
    """
    SCALAR = 1
    VECTOR = 2
    TENSOR = 3


class MeshEntityType(Enum):
    """
    MeshEntityType is an Enum describing Types in Mesh topology where postprocessing data can be associated with.
    """
    NODE = 1
    ELEMENT = 2


# XDMFDICT desribes a dict that converts element shapes to xdmf data attributes
XDMFDICT = {'Tri3': {'no_of_nodes': 3, 'xdmf_name': 'Triangle'},
            'Tri6': {'no_of_nodes': 6, 'xdmf_name': 'Triangle_6'},
            'Quad4': {'no_of_nodes': 4, 'xdmf_name': 'Quadrilateral'},
            'Quad8': {'no_of_nodes': 8, 'xdmf_name': 'Quadrilateral_8'},
            'Hexa8': {'no_of_nodes': 8, 'xdmf_name': 'Hexahedron'},
            'Hexa20': {'no_of_nodes': 20, 'xdmf_name': 'Hexahedron_20'},
            'Tet4': {'no_of_nodes': 4, 'xdmf_name': 'Tetrahedron'},
            'Tet10': {'no_of_nodes': 10, 'xdmf_name': 'Tetrahedron_10'},
            'straight_line': {'no_of_nodes': 2, 'xdmf_name': 'Polyline'},
            'quadratic_line': {'no_of_nodes': 3, 'xdmf_name': 'Edge_3'},
            'point': {'no_of_nodes': 1, 'xdmf_name': 'Polyvertex'},
            }

# Describe Conversion from Mesh Entity Enum to XDMF type
MESHENTITYTYPE2XDMF = {MeshEntityType.NODE: 'Node',
                       MeshEntityType.ELEMENT: 'Cell'
                       }

# Describe Conversion from Post Process Data Type Enum to XDMF type
POSTPROCESSDATATYPE2XDMF = {PostProcessDataType.SCALAR: 'Scalar',
                            PostProcessDataType.VECTOR: 'Vector',
                            PostProcessDataType.TENSOR: 'Tensor'
                            }

# Describe dimension from PostProcessDataType for XDMF Files
POSTPROCESSDATATYPE2XDMFDIMENSION = {PostProcessDataType.SCALAR: 1,
                                     PostProcessDataType.VECTOR: 3,
                                     PostProcessDataType.TENSOR: 9,
                                     }
