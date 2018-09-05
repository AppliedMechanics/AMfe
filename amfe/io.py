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
import numpy as np

from amfe import Mesh

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
    Reads json-ascii-files created by the GiD-AMfe-extension.
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
        Parse the GiD-json-file to the object specified by the builder (MeshConverter object).
        
        Parameters
        ----------
        verbose : bool

        Returns
        -------
        object
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

            print("Import nodes...")
            for counter, node in enumerate(json_tree['nodes']):
                self.builder.build_node(node['id'], node['coords'][0], node['coords'][1], node['coords'][2])
                print("\rImport node no. {} / {}".format(counter, no_of_nodes), end='')

            print("\n...finished")
            print("Import elements")
            for ele_type in json_tree['elements']:
                current_amfe_eletype = self.eletypes[(ele_type['ele_type'], json_tree['quadratic'])]
                print("    Import eletype {} ...".format(current_amfe_eletype))
                for counter, element in enumerate(ele_type['elements']):
                    eleid = element['id']
                    nodes = element['connectivity'][:-1]
                    self.builder.build_element(eleid, current_amfe_eletype, nodes)
                    print("\rImport element No. {} / {}".format(counter, no_of_elements), end='')
                print("\n    ...finished")
            print("\n...finished")

            print("Import groups...")
            for group in json_tree['groups']:
                self.builder.build_group(group, nodeids=json_tree['groups'][group]['nodes'],
                                         elementids=json_tree['groups'][group]['elements'])
            print("...finished")

        # Finished build, return mesh
        return self.builder.return_mesh()


class GmshAsciiMeshReader(MeshReader):

    eletypes = {
        1: 'straight_line',
        2: 'Tri3',
        3: 'Quad4',
        4: 'Tet4',
        5: 'Hex8',
        6: 'Prism6',
        7: None,  # Pyramid
        8: 'quadratic_line',
        9: 'Tri6',
        10: None,  # 9 node quad
        11: 'Tet10',
        12: None,  # 27 Node Hex
        13: None,  # 2nd order prism
        14: None,  # 2nd order pyramid
        15: 'point',
        16: 'Quad8',
        17: 'Hex20',
        18: None,  # 15node 2nd order prism
        19: None,  # 13 node pyramid
        20: None,  # 9 node triangle
        21: 'Tri10',
        22: None,  # 12 node triangle
        23: None,  # 15 node triangle
        24: None,
        25: None,
        26: None,
        27: None,
        28: None,
        29: None,
        30: None,
        31: None,
        92: None,
        93: None
    }

    eletypes_3d = [4, 5, 6, 11, 17]

    tag_format_start = "$MeshFormat"
    tag_format_end = "$EndMeshFormat"
    tag_nodes_start = "$Nodes"
    tag_nodes_end = "$EndNodes"
    tag_elements_start = "$Elements"
    tag_elements_end = "$EndElements"
    tag_physical_names_start = "$PhysicalNames"
    tag_physical_names_end = "$EndPhysicalNames"

    def __init__(self, filename=None, builder=None):
        super().__init__()
        self._builder = builder
        self._filename = filename
        # Default dimension is 2, later (during build of elements) it is checked if mesh is 3D
        self._dimension = 2

    def parse(self):
        with open(self._filename, 'r') as infile:
            # Read all lines into data_geometry
            data_geometry = infile.read().splitlines()

        n_nodes = None
        n_elements = None
        i_nodes_start = None
        i_nodes_end = None
        i_elements_start = None
        i_elements_end = None
        i_format_start = None
        i_format_end = None
        i_physical_names_start = None
        i_physical_names_end = None

        # Store indices of lines where different sections start and end
        for s in data_geometry:
            if s == self.tag_format_start:  # Start Formatliste
                i_format_start = data_geometry.index(s) + 1
            elif s == self.tag_format_end:  # Ende Formatliste
                i_format_end = data_geometry.index(s)
            elif s == self.tag_nodes_start:  # Start Knotenliste
                i_nodes_start = data_geometry.index(s) + 2
                n_nodes = int(data_geometry[i_nodes_start - 1])
            elif s == self.tag_nodes_end:  # Ende Knotenliste
                i_nodes_end = data_geometry.index(s)
            elif s == self.tag_elements_start:  # Start Elementliste
                i_elements_start = data_geometry.index(s) + 2
                n_elements = int(data_geometry[i_elements_start - 1])
            elif s == self.tag_elements_end:  # Ende Elementliste
                i_elements_end = data_geometry.index(s)
            elif s == self.tag_physical_names_start:  # Start Physical Names
                i_physical_names_start = data_geometry.index(s) + 2
            elif s == self.tag_physical_names_end:
                i_physical_names_end = data_geometry.index(s)

        # build number of nodes and elements
        if n_nodes is not None and n_elements is not None:
            self._builder.build_no_of_nodes(n_nodes)
            self._builder.build_no_of_elements(n_elements)
        else:
            raise ValueError('Could not read number of nodes and number of elements in File {}'.format(self._filename))

        # Check if indices could be read:
        for i in [i_nodes_start, i_nodes_end, i_elements_start, i_elements_end, i_format_start, i_format_end]:
            if i is None:
                raise ValueError('Could not read start and end tags of format, nodes and elements '
                                 'in file {}'.format(self._filename))

        # Check inconsistent dimensions
        if (i_nodes_end - i_nodes_start) != n_nodes \
                or (i_elements_end - i_elements_start) != n_elements:
            raise ValueError('Error while processing the file {}',
                             'Dimensions are not consistent.'.format(self._filename))

        # extract data from file to lists
        list_imported_mesh_format = data_geometry[i_format_start: i_format_end]
        list_imported_nodes = data_geometry[i_nodes_start: i_nodes_end]
        list_imported_elements = data_geometry[i_elements_start: i_elements_end]

        # Dict for physical names
        groupnames = dict()
        if i_physical_names_start is not None and i_physical_names_end is not None:
            list_imported_physical_names = data_geometry[i_physical_names_start: i_physical_names_end]
            # Make a dict for physical names:
            for group in list_imported_physical_names:
                groupinfo = group.split()
                idx = int(groupinfo[1])
                # split double quotes
                name = groupinfo[2][1:-1]
                groupnames.update({idx: name})

        # conversion of the read strings in mesh format to integer and floats
        for j in range(len(list_imported_mesh_format)):
            list_imported_mesh_format[j] = [float(x) for x in
                                            list_imported_mesh_format[j].split()]

        # Build nodes
        for node in list_imported_nodes:
            nodeinfo = node.split()
            nodeid = int(nodeinfo[0])
            x = float(nodeinfo[1])
            y = float(nodeinfo[2])
            z = float(nodeinfo[3])
            self._builder.build_node(nodeid, x, y, z)

        # Build elements
        groupentities = dict()
        for element in list_imported_elements:
            elementinfo = element.split()
            elementid = int(elementinfo[0])
            eletype = self.eletypes[int(elementinfo[1])]
            if int(elementinfo[1]) in self.eletypes_3d:
                self._dimension = 3
            no_of_tags = int(elementinfo[2])
            physical_group = int(elementinfo[3])
            connectivity = elementinfo[2+no_of_tags+1:]
            connectivity = [int(node) for node in connectivity]

            self._builder.build_element(elementid, eletype, connectivity)
            # Add element to group
            if physical_group in groupentities:
                groupentities[physical_group].append(elementid)
            else:
                groupentities.update({physical_group: [elementid]})

        # Build groups
        for group in groupentities:
            if group in groupnames:
                self.builder.build_group(groupnames[group], [], groupentities[group])
            else:
                self.builder.build_group(group, [], groupentities[group])

        self.builder.build_mesh_dimension(self._dimension)

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

    def build_group(self, name, nodeids, elementids):
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


class AmfeMeshConverter(MeshConverter):
    """
    Converter for AMfe meshes.

    Examples
    --------
    Convert a GiD json file to an AMfe mesh:

    >>> from amfe.io import GidJsonMeshReader, AmfeMeshConverter
    >>> filename = '/path/to/your/file.json'
    >>> converter = AmfeMeshConverter()
    >>> reader = GidJsonMeshReader(filename, converter)
    >>> mymesh = reader.parse()

    """

    element_2d_set = {'Tri6', 'Tri3', 'Quad4', 'Quad8', }
    element_3d_set = {'Tet4', 'Tet10', 'Hexa8', 'Hexa20', 'Prism6'}

    boundary_2d_set = {'straight_line', 'quadratic_line'}
    boundary_3d_set = {'straight_line', 'quadratic_line', 'Tri6', 'Tri3', 'Tri10', 'Quad4', 'Quad8'}

    def __init__(self, verbose=False):
        super().__init__()
        self._verbose = verbose
        self._mesh = Mesh()
        self._dimension = None
        self._no_of_nodes = None
        self._no_of_elements = None
        self._nodes = np.empty((0, 3), dtype=float)
        self._currentnodeid = 0
        self._currentelementid = 0
        self._connectivity = list()
        self._eleshapes = list()
        self._groups = dict()
        # mapping from reader-nodeid to amfe-nodeid
        self._nodeid2idx = dict()
        self._elementid2idx = dict()
        return

    def build_no_of_nodes(self, no):
        # This function is only used for preallocation
        # It is not necessary to call, but useful if information about no_of_nodes exists
        self._no_of_nodes = no
        if self._nodes.shape[0] == 0:
            self._nodes = np.zeros((no, 3), dtype=float)
        return

    def build_no_of_elements(self, no):
        # This function is not used
        # If someone wants to improve performance he/she can add preallocation functionality for elements
        self._no_of_elements = no
        # preallocation...
        return

    def build_mesh_dimension(self, dim):
        self._dimension = dim
        return

    def build_node(self, idx, x, y, z):
        # amfeid is the row-index in nodes array
        amfeid = self._currentnodeid
        # Check if preallocation has been done so far
        if self._no_of_nodes is not None:
            # write node in preallocated array
            self._nodes[amfeid, :] = [x, y, z]
        else:
            # append node if array is not preallocated with full node dimension
            self._nodes = np.append(self._nodes, np.array([x, y, z], dtype=float, ndmin=2), axis=0)
        # add mapping information
        self._nodeid2idx.update({idx: amfeid})
        # increment row-index counter
        self._currentnodeid += 1
        return

    def build_element(self, idx, etype, nodes):
        # append connectivity information
        self._connectivity.append(np.array(nodes, dtype=int))
        # update mapping information
        self._elementid2idx.update({idx: self._currentelementid})
        # append element type information
        # Hint: The differentiation between volume and boundary elements will be performed after all
        # elements have been read
        self._eleshapes.append(etype)
        # increment row-index counter
        self._currentelementid += 1
        return

    def build_group(self, name, nodeids=None, elementids=None):
        # append group information
        group = {name: {'nodes': nodeids, 'elements': elementids}}
        self._groups.update(group)
        return

    def return_mesh(self):
        # Check dimension of model
        if self._dimension is None:
            if not self.element_3d_set.intersection(set(self._eleshapes)):
                # No 3D element in eleshapes, thus:
                self._dimension = 2
            else:
                self._dimension = 3
        # If dimension = 2 cut the z coordinate
        if self._dimension == 2:
            self._mesh.nodes = self._nodes[:, :self._dimension]
        # set the node mapping information
        self._mesh.nodeid2idx = self._nodeid2idx
        # divide in boundary and volume elements
        currentidx = 0
        currentboundaryidx = 0
        elementid2idx = dict()
        if self._dimension == 3:
            volume_element_set = self.element_3d_set
            boundary_element_set = self.boundary_3d_set
        elif self._dimension == 2:
            volume_element_set = self.element_2d_set
            boundary_element_set = self.boundary_2d_set
        else:
            raise ValueError('Dimension must be 2 or 3')

        # write properties
        self._mesh.dimension = self._dimension
        self._mesh.connectivity = [np.array([self._nodeid2idx[nodeid] for nodeid in element])
                                   for index, element in enumerate(self._connectivity)
                                   if self._eleshapes[index] in volume_element_set]
        self._mesh.boundary_connectivity = [np.array([self._nodeid2idx[nodeid] for nodeid in element])
                                            for index, element in enumerate(self._connectivity)
                                            if self._eleshapes[index] in boundary_element_set]
        self._mesh.ele_shapes = [eleshape
                                 for index, eleshape in enumerate(self._eleshapes)
                                 if self._eleshapes[index] in volume_element_set]
        self._mesh.boundary_ele_shapes = [eleshape
                                          for index, eleshape in enumerate(self._eleshapes)
                                          if self._eleshapes[index] in boundary_element_set]
        for eleid in self._elementid2idx:
            if self._eleshapes[self._elementid2idx[eleid]] in volume_element_set:
                elementid2idx.update({eleid: (0, currentidx)})
                currentidx += 1
            elif self._eleshapes[self._elementid2idx[eleid]] in boundary_element_set:
                elementid2idx.update({eleid: (1, currentboundaryidx)})
                currentboundaryidx += 1
        self._mesh.eleid2idx = elementid2idx
        self._mesh.groups = self._groups
        return self._mesh
