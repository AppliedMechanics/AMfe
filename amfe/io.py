# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
io module of AMfe

It handles input output operations for AMfe
"""

import abc
import re
import os
import json

__all__ = [
    'MeshReader',
    'MeshConverter',
    'GidAsciiMeshReader',
    'GidJsonMeshReader',
]


def check_dir(*filenames):
    """
    Check if paths exists; if not, the given paths will be created.

    Parameters
    ----------
    *filenames : string or list of strings
        string containing a path.

    Returns
    -------
    None
    """
    for filename in filenames:  # loop on files
        dir_name = os.path.dirname(filename)
        # check if directory does not exist; then create directory
        if not os.path.exists(dir_name) or dir_name == '':
            os.makedirs(os.path.dirname(filename))  # then create directory
            print("Created directory: " + os.path.dirname(filename))


class MeshReader(abc.ABC):
    """
    Abstract super class for all MeshReaders.

    The tasks of the MeshReaders are:
    ---------------------------------

    - Read line by line a stream (or file)
    - Call MeshConverter function for each line

    PLEASE FOLLOW THE BUILDER PATTERN!
    """

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


class GidAsciiMeshReader(MeshReader):
    """
    Reads GID-Ascii-Files
    """

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

    def parse(self, verbose=False):
        with open(self._filename, 'r') as infile:
            line = next(infile)
            pattern = "dimension (\d) ElemType\s([A-Za-z0-9]*)\sNnode\s(\d)"
            match = re.search(pattern, line)
            dimension = int(match.group(1))  # dimension (nodes have two or three coordinates)
            eleshape = match.group(2)  # elementtype
            nnodes = int(match.group(3))  # number of nodes per element

            self.builder.build_mesh_dimension(dimension)
            try:
                eletype = self.eletypes[(eleshape, nnodes)]
            except Exception:
                print('Eletype ({},{})  cannot be found in eletypes dictionary, it is not implemented in AMfe'.format(
                    eleshape, nnodes))
                return
            if eleshape is None:
                raise ValueError('Element ({},{}) is not implemented in AMfe'.format(eleshape, nnodes))
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


class GidJsonMeshReader(MeshReader):
    '''
    Reads Json-Ascii-Files created by GID AMfe Extension
    '''

    # Eletypes dict:
    # {('shape', 0/1 (=non-quadratic/quadratic)): 'AMfe-name'}
    eletypes = {
        ('Line', 0): 'straight_line',
        ('Line', 1): 'quadratic_line',
        ('Triangle', 0): 'Tri3',
        ('Triangle', 1): 'Tri6',
        ('Triangle', 2): 'Tri10',
        ('Quadrilateral', 0): 'Quad4',
        ('Quadrilateral', 1): 'Quad8',
        ('Tetrahedra', 0): 'Tet4',
        ('Tetrahedra', 1): 'Tet10',
        ('Hexahedra', 0): 'Hexa8',
        ('Hexahedra', 1): 'Hexa20',
        ('Prism', 0): 'Prism6',
        ('Prism', 1): None,
        ('Pyramid', 0): None,
        ('Pyramid', 1): None,
        ('Point', 0): 'point',
        ('Point', 1): 'point',
        ('Sphere', 0): None,
        ('Sphere', 1): None,
        ('Circle', 0): None,
        ('Circle', 1): None
    }

    eletypes_3d = {'Tetrahedra', 'Hexahedra', 'Prism', 'Pyramid'}

    def __init__(self, filename=None, builder=None):
        super().__init__()
        self._filename = filename
        self._builder = builder

    def parse(self, verbose=False):
        """
        Parse the GidJsonFile to the object specified by the builder (MeshConverter object)
        
        Parameters
        ----------
        verbose

        Returns
        -------
        Object
        """
        with open(self._filename, 'r') as infile:
            json_tree = json.load(infile)

            dimflag = set([ele_type['ele_type'] for ele_type in json_tree['elements']]).intersection(self.eletypes_3d)
            if not dimflag:
                self.builder.build_mesh_dimension(2)
            else:
                self.builder.build_mesh_dimension(3)

            no_of_nodes = json_tree['no_of_nodes']
            no_of_elements = json_tree['no_of_elements']

            self.builder.build_no_of_nodes(no_of_nodes)
            self.builder.build_no_of_elements(no_of_elements)

            print("Import Nodes...")
            for counter, node in enumerate(json_tree['nodes']):
                self.builder.build_node(node['id'], node['coords'][0], node['coords'][1], node['coords'][2])
                print("\rImport Node No. {} / {}".format(counter, no_of_nodes), end='')

            print("\n...finished")
            print("Import Elements")
            for ele_type in json_tree['elements']:
                current_amfe_eletype = self.eletypes[(ele_type['ele_type'], json_tree['quadratic'])]
                print("    Import Eletype {} ...".format(current_amfe_eletype))
                for counter, element in enumerate(ele_type['elements']):
                    eleid = element['id']
                    nodes = element['connectivity'][:-1]
                    self.builder.build_element(eleid, current_amfe_eletype, nodes)
                    print("\rImport Element No. {} / {}".format(counter, no_of_elements), end='')
                print("\n    ...finished")
            print("\n...finished")

            print("Import Groups...")
            for group in json_tree['groups']:
                self.builder.build_group(group, nodeids=json_tree['groups'][group]['nodes'],
                                         elementids=json_tree['groups'][group]['elements'])
            print("...finished")
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
        name: string
            name identifying the node group
        nodeids: list
            list with node ids
        elementids: list
            list with element ids

        Returns
        -------

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
