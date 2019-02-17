#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Gmsh ascii mesh reader for I/O module.
"""

from amfe.io.mesh.base import MeshReader

__all__ = [
    'GmshAsciiMeshReader'
    ]


class GmshAsciiMeshReader(MeshReader):
    """
    Reader for gmsh ascii files.
    """

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

    def __init__(self, filename=None):
        super().__init__()
        self._filename = filename
        # Default dimension is 2, later (during build of elements) it is checked if mesh is 3D
        self._dimension = 2
        return

    def parse(self, builder):
        """
        Parse the Mesh with builder

        Parameters
        ----------
        builder : MeshConverter
            Mesh converter object that builds the mesh

        Returns
        -------
        None
        """
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
        for index, s in enumerate(data_geometry):
            if s == self.tag_format_start:  # Start Formatliste
                i_format_start = index + 1
            elif s == self.tag_format_end:  # Ende Formatliste
                i_format_end = index
            elif s == self.tag_nodes_start:  # Start Knotenliste
                i_nodes_start = index + 2
                n_nodes = int(data_geometry[i_nodes_start - 1])
            elif s == self.tag_nodes_end:  # Ende Knotenliste
                i_nodes_end = index
            elif s == self.tag_elements_start:  # Start Elementliste
                i_elements_start = index + 2
                n_elements = int(data_geometry[i_elements_start - 1])
            elif s == self.tag_elements_end:  # Ende Elementliste
                i_elements_end = index
            elif s == self.tag_physical_names_start:  # Start Physical Names
                i_physical_names_start = index + 2
            elif s == self.tag_physical_names_end:
                i_physical_names_end = index

        # build number of nodes and elements
        if n_nodes is not None and n_elements is not None:
            builder.build_no_of_nodes(n_nodes)
            builder.build_no_of_elements(n_elements)
        else:
            raise ValueError('Could not read number of nodes and number of elements in File {}'.format(self._filename))

        # Check if indices could be read:
        if None in [i_nodes_start, i_nodes_end, i_elements_start, i_elements_end, i_format_start, i_format_end]:
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
            builder.build_node(nodeid, x, y, z)

        # Build elements
        groupentities = dict()
        elemental_tags = dict()
        tag_entities = {'no_of_mesh_partitions': {},
                        'partition_id': {},
                        'partitions_neighbors': {},
                        }
        has_partitions = False
        for element in list_imported_elements:
            elementinfo = element.split()
            elementid = int(elementinfo[0])
            eletype = self.eletypes[int(elementinfo[1])]
            if int(elementinfo[1]) in self.eletypes_3d:
                self._dimension = 3
            no_of_tags = int(elementinfo[2])
            connectivity = elementinfo[2+no_of_tags+1:]
            connectivity = [int(node) for node in connectivity]

            builder.build_element(elementid, eletype, connectivity)
            # Add element to group
            physical_group = int(elementinfo[3])
            elemental_tag = int(elementinfo[4])
            if physical_group in groupentities:
                groupentities[physical_group].append(elementid)
            else:
                groupentities.update({physical_group: [elementid]})
            if elemental_tag in elemental_tags:
                elemental_tags[elemental_tag].append(elementid)
            else:
                elemental_tags.update({elemental_tag: [elementid]})
            
            # add element to tags 
            if no_of_tags>3: 
                has_partitions = True               
                tag_list = [int(tag) for tag in elementinfo[3:3+no_of_tags][2:]]
                if no_of_tags==3:
                    tag_list.extend([None,None])
                elif no_of_tags==4:
                    tag_list.extend([None])
                
                no_of_mesh_partitions, partition_id, *partition_neighbors = tag_list
                elem_tag_dict = {'no_of_mesh_partitions' : no_of_mesh_partitions, 
                                 'partition_id' : partition_id, 
                                 'partitions_neighbors' : tuple(partition_neighbors)}
                for tag_name, dict_tag in tag_entities.items():
                    elem_tag_value = elem_tag_dict[tag_name]
                    if elem_tag_value in dict_tag:
                        dict_tag[elem_tag_value].append(elementid)
                    else:
                        dict_tag[elem_tag_value] = [elementid]

        # Build groups
        for group in groupentities:
            if group in groupnames:
                builder.build_group(groupnames[group], [], groupentities[group])
            else:
                builder.build_group(group, [], groupentities[group])

        # Build tags
        if has_partitions:
            tags_dict = tag_entities
            tags_dict.update({'elemental_group': elemental_tags})
        else:
            tags_dict = {'elemental_group': elemental_tags}
        builder.build_tag(tags_dict)

        builder.build_mesh_dimension(self._dimension)

        return