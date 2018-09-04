# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

"""
I/O module of AMfe.

It handles input/output operations for AMfe.
"""

import abc
import re
import os

__all__ = [
    'MeshReader',
    'MeshConverter',
    'GidAsciiMeshReader'
]


def check_dir(*filenames):
    '''
    Check if path(s) exist; if not, given path(s) will be created.

    Parameters
    ----------
    *filenames : str or list of str
        String or list of strings containing path(s).

    Returns
    -------
    None
    '''

    for filename in filenames:  # loop over files
        dir_name = os.path.dirname(filename)
        # check whether directory does exist
        if not os.path.exists(dir_name) or dir_name == '':
            os.makedirs(os.path.dirname(filename))  # if not, then create directory
            print('Created directory \'' + os.path.dirname(filename) + '\'.')
    return


class MeshReader(abc.ABC):
    '''
    Abstract super class for all MeshReaders.

    Tasks of the MeshReaders:
    -------------------------
    - Read line by line a stream (or file).
    - Call MeshConverter function for each line.

    Notes:
    -----
    PLEASE FOLLOW THE BUILDER PATTERN!
    '''

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        self._builder = None
        return

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
            raise ValueError('Invalid builder given.')
        return


class GidAsciiMeshReader(MeshReader):
    '''
    Reads GiD-Ascii-files.
    '''

    eletypes = {
        ('Linear', 2): 'straight_line',
        ('Linear', 3): 'quadratic_line',
        ('Triangle', 3): 'Tri3',
        ('Triangle', 6): 'Tri6',
        ('Triangle', 10): 'Tri10',
        ('Quadrilateral', 4): 'Quad4',
        ('Quadrilateral', 8): 'Quad8',
        ('Tetrahedra', 4): 'Tet4',
        ('Tetrahedra', 10): 'Tet10',
        ('Hexahedra', 8): 'Hexa8',
        ('Hexahedra', 20): 'Hexa20',
        ('Prism', 6): 'Prism6',
        ('Pyramid', 6): None,
        ('Point', 1): 'point',
        ('Sphere', -1): None,
        ('Circle', -1): None,
    }

    def __init__(self, filename=None, builder=None):
        super().__init__()
        self._filename = filename
        self.builder = builder
        return

    def parse(self, verbose=False):
        with open(self._filename, 'r') as infile:
            line = next(infile)
            pattern = 'dimension (\d) ElemType\s([A-Za-z0-9]*)\sNnode\s(\d)'
            match = re.search(pattern, line)
            dimension = int(match.group(1))  # dimension (nodes have 2 or 3 coordinates)
            eleshape = match.group(2)  # elementtype
            nnodes = int(match.group(3))  # number of nodes per element

            self.builder.build_mesh_dimension(dimension)
            try:
                eletype = self.eletypes[(eleshape, nnodes)]
            except Exception:
                print('Eletype ({},{}) cannot be found in eletypes dictionary, it is not implemented in AMfe.'
                      .format( eletype, nnodes))
            if eletype is None:
                raise ValueError('Element ({},{}) is not implemented in AMfe.'.format(eletype, nnodes))
            if verbose:
                print('Eletype {} identified.'.format(eletype))

            # Coordinates
            for line in infile:
                if line.strip() == 'Coordinates':
                    for line in infile:
                        try:
                            nodeid = int(line[0:5])
                            x = float(line[5:21])
                            y = float(line[21:37])
                            z = float(line[37:53])
                        except ValueError:
                            if line.strip() == "End Coordinates":
                                break
                            else:
                                raise
                        self.builder.build_node(nodeid, x, y, z)

                elif line.strip() == 'Elements':
                    for line in infile:
                        try:
                            element = [int(e) for e in line.split()]
                            eleid = element[0]
                            nodes = element[1:]
                        except ValueError:
                            if line.strip() == "End Elements":
                                break
                            else:
                                raise
                        self.builder.build_element(eleid, eletype, nodes)
                else:
                    print(line)

        # Finished build, return mesh
        return self.builder.return_mesh()


class MeshConverter:
    '''
    Super class for all MeshConverters.
    '''

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

    def build_group(self, name, nodeids=None, elementids=None):
        '''

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
        '''
        pass

    def build_material(self, material):
        pass

    def build_partition(self, partition):
        pass

    def build_mesh_dimension(self, dim):
        pass

    def return_mesh(self):
        pass
