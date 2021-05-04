#
# Copyright (c) 2021 TECHNICAL UNIVERSITY OF MUNICH,
# DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license.
# See LICENSE file for more information.
#

"""
Gmsh ascii mesh reader for File Format Version 4 for I/O module.
"""

import numpy as np
from amfe.io.mesh.base import MeshReader

__all__ = [
    'GmshAscii4MeshReader'
]


class GmshAscii4MeshReader(MeshReader):
    """Reader for gmsh ascii version 4 files.

    This is a MeshReader for the Gmsh ASCII File Format Version 4.

    It can import nodes, elements, partition-ids, entity ids and their
    dimension, and the physical names.

    The entities are stored in two different tags according to the structure
    of the gmsh file format.
    The first tag is 'gmsh_entity_dimension' which stores the spatial
    dimension of the entity.
    The second tag is 'gmsh_entity_id' which stores an ID for the entity.
    Please note: The gmsh_entity_id is not unique. One always has to combine
    the gmsh_entity_dimension and the gmsh_entity_id to get a unique
    identifier.
    Example: There can be an entity with ID 5 for a curve objects and an
    entity with the same id for surface objects. One can only distinguish them
    by the tuple (gmsh_entity_dimension, gmsh_entity_id). The first tuple
    would be (1, 5) and the second would be (2, 5) because curves have
    spatial dimension 1 while surfaces have spatial dimension 2.

    Partition IDs are stored in the Tag 'gmsh_partition_ids'.
    It stores the partion ids the elements belong to in tuples because one
    element can belong to several partitions.

    Physical Names are converted to groups. The groups keys are the physical
    names. Their values are the Element-IDs. Only those element ids are added
    to the group, which have the same dimension as the dimension of the
    physical name that is specified in the Gmsh file section $PhysicalNames.
    This is considered more user-friendly than adding elements from other
    dimensions but with same physical tags that were assigned to the
    Physical Name.
    If the Physical Name has dimension zero, nodes are added to the group, too.
    """

    _eletypes = {
        1: 'straight_line',
        2: 'Tri3',
        3: 'Quad4',
        4: 'Tet4',
        5: 'Hexa8',
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
        17: 'Hexa20',
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

    _eletypes_3d = ['Tet4', 'Hexa8', 'Prism6', 'Tet10', 'Hexa20']

    _tag_format_start = "$MeshFormat"
    _tag_format_end = "$EndMeshFormat"
    _tag_entities_start = "$Entities"
    _tag_entities_end = "$EndEntities"
    _tag_nodes_start = "$Nodes"
    _tag_nodes_end = "$EndNodes"
    _tag_elements_start = "$Elements"
    _tag_elements_end = "$EndElements"
    _tag_physical_names_start = "$PhysicalNames"
    _tag_physical_names_end = "$EndPhysicalNames"
    _tag_partitioned_entities_start = "$PartitionedEntities"
    _tag_partitioned_entities_end = "$EndPartitionedEntities"

    def __init__(self, filename=None):
        super().__init__()
        self._filename = filename
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
        i_entities_start = None
        i_entities_end = None
        i_nodes_start = None
        i_nodes_end = None
        i_elements_start = None
        i_elements_end = None
        i_format_start = None
        i_format_end = None
        i_physical_names_start = None
        i_physical_names_end = None
        i_partitioned_entities_start = None
        i_partitioned_entities_end = None

        has_partitions = False

        n_entity_blocks_for_nodes = 0
        n_entity_blocks_for_elements = 0

        # Step 1:
        # Store indices of lines where different sections start and end
        for index, s in enumerate(data_geometry):
            if s == self._tag_nodes_start:  # Start Node List
                i_nodes_start = index + 1
                info_nodes = data_geometry[i_nodes_start].split()
                n_nodes = int(info_nodes[1])
                n_entity_blocks_for_nodes = int(info_nodes[0])
            elif s == self._tag_nodes_end:  # End Node List
                i_nodes_end = index
            elif s == self._tag_elements_start:  # Start Element List
                i_elements_start = index + 1
                info_elements = data_geometry[i_elements_start].split()
                n_elements = int(info_elements[1])
                n_entity_blocks_for_elements = int(info_elements[0])
            elif s == self._tag_elements_end:  # End Element List
                i_elements_end = index
            elif s == self._tag_physical_names_start:  # Start Physical Names
                i_physical_names_start = index + 1
            elif s == self._tag_physical_names_end:  # End Physical Names
                i_physical_names_end = index
            elif s == self._tag_format_start:  # Start Format
                i_format_start = index + 1
            elif s == self._tag_format_end:  # End Format
                i_format_end = index
            elif s == self._tag_entities_start:  # Start Entities
                i_entities_start = index + 1
            elif s == self._tag_entities_end:  # End Entities
                i_entities_end = index
            elif s == self._tag_partitioned_entities_start:
                i_partitioned_entities_start = index + 1
            elif s == self._tag_partitioned_entities_end:
                i_partitioned_entities_end = index

        if (i_partitioned_entities_start is not None and
                i_partitioned_entities_end is not None):
            has_partitions = True
            num_ghost_entities = int(
                data_geometry[i_partitioned_entities_start + 1].split()[0])
            update_entities_start =\
                i_partitioned_entities_start + 2 + num_ghost_entities
            info_entities = data_geometry[update_entities_start].split()
        else:
            info_entities = data_geometry[i_entities_start].split()
            update_entities_start = i_entities_start

        n_points = int(info_entities[0])
        n_curves = int(info_entities[1])
        n_surfaces = int(info_entities[2])
        n_volumes = int(info_entities[3])

        # Step 0:
        # build number of nodes and elements
        if n_nodes is not None and n_elements is not None:
            builder.build_no_of_nodes(n_nodes)
            builder.build_no_of_elements(n_elements)
        else:
            raise ValueError('Could not read number of nodes and number'
                             'of elements in File {}'.format(self._filename))

        # Check Consistency of File
        if None in [i_nodes_start, i_nodes_end, i_elements_start,
                    i_elements_end, i_format_start, i_format_end,
                    i_entities_start, i_entities_end]:
            raise ValueError('Could not read start and end tags of format,'
                             'entities, nodes and elements '
                             'in file {}'.format(self._filename))

        # Check inconsistent dimensions
        if int((i_nodes_end - i_nodes_start - n_entity_blocks_for_nodes - 1))\
                != 2*n_nodes \
                or (i_elements_end - i_elements_start -
                    n_entity_blocks_for_elements - 1) != n_elements:
            raise ValueError('Error while processing the file {}',
                             'Dimensions are not consistent.'.format(
                                 self._filename))

        # Step 2:
        # Read Physical Names
        dim2phytag2groupname = dict()
        if (i_physical_names_start is not None
                and i_physical_names_end is not None):
            list_imported_physical_names = data_geometry[
                                           i_physical_names_start + 1:
                                           i_physical_names_end]
            # Make a dict for physical names:
            for group in list_imported_physical_names:
                groupinfo = group.split()
                dim = int(groupinfo[0])
                idx = int(groupinfo[1])
                # split double quotes
                name = groupinfo[2][1:-1]
                if dim in dim2phytag2groupname.keys():
                    dim2phytag2groupname[dim].update({idx: name})
                else:
                    dim2phytag2groupname.update({dim: {idx: name}})

        # Step 3:
        # Read Entities or PartitionedEntities, respectively
        dim2entities2phytags = dict()
        dim2entities2partitions = dict()
        dim2phytag2entity = dict({0: {}, 1: {}, 2: {}, 3: {}})

        def update_entities(n_of_entities, start, spatial_dim):
            ent_dict = dict()
            part_dict = dict()
            for e in range(n_of_entities):
                entities_info = data_geometry[start + 1 + e].split()
                enti_id = int(entities_info[0])
                shift = 0
                if has_partitions:
                    no_of_partitions = int(entities_info[3])
                    partitions = [int(x) for x in
                                  entities_info[4:4 + no_of_partitions]]
                    part_dict.update({enti_id: partitions})
                    shift = 3 + no_of_partitions
                if spatial_dim == 0:
                    no_of_physical_tags = int(entities_info[4 + shift])
                    physical_tags = \
                        [int(x) for x in entities_info[
                                         5 + shift:
                                         5 + shift + no_of_physical_tags]]
                else:
                    no_of_physical_tags = int(entities_info[7 + shift])
                    physical_tags = \
                        [int(x) for x in entities_info[
                                         8 + shift:
                                         8 + shift + no_of_physical_tags]]
                for phytag in physical_tags:
                    if phytag in dim2phytag2entity[spatial_dim]:
                        dim2phytag2entity[spatial_dim][phytag].append(enti_id)
                    else:
                        dim2phytag2entity[spatial_dim].update(
                            {phytag: [enti_id]})

                ent_dict.update({enti_id: physical_tags})

            dim2entities2phytags.update({spatial_dim: ent_dict})
            dim2entities2partitions.update({spatial_dim: part_dict})

        update_entities(n_points, update_entities_start, 0)
        update_entities(n_curves, update_entities_start + n_points, 1)
        update_entities(n_surfaces, update_entities_start + n_points +
                        n_curves, 2)
        update_entities(n_volumes, update_entities_start + n_points +
                        n_curves + n_surfaces, 3)

        # Step 4:
        # Extract nodes information and build nodes
        entity2nodeid = dict()
        index = i_nodes_start + 1
        for i in range(0, n_entity_blocks_for_nodes):
            info_entity = data_geometry[index].split()
            entity_id = int(info_entity[1])
            n_nodes_entity = int(info_entity[3])
            n_list = np.zeros(n_nodes_entity, dtype=int)
            for j in range(n_nodes_entity):
                nodeid = int(data_geometry[index + 1 + j])
                nodecoords = [float(x) for x in data_geometry[
                    index + 1 + j + n_nodes_entity].split()]
                builder.build_node(nodeid, *nodecoords)
                n_list[j] = nodeid
            index = index + 2 * n_nodes_entity + 1
            entity2nodeid.update({entity_id: n_list})

        # Step 5:
        # Extract elements information and build elements
        dim2entity2eleid = dict()
        index = i_elements_start + 1
        for i in range(0, n_entity_blocks_for_elements):
            info_entity = data_geometry[index].split()
            entity_dimension = int(info_entity[0])
            entity_id = int(info_entity[1])
            element_type = int(info_entity[2])
            shape = self._eletypes[element_type]
            n_ele_in_current_block = int(info_entity[3])
            e_list = np.zeros(n_ele_in_current_block, dtype=int)

            for j in range(n_ele_in_current_block):
                ele_info = [int(x) for x in data_geometry[
                    index + 1 + j].split()]
                ele_id = ele_info[0]
                connectivity = np.array(ele_info[1:], dtype=int)
                e_list[j] = ele_id

                if shape in self._eletypes_3d:
                    self._dimension = 3

                # Change the indices of Tet10-elements, as they are numbered
                # differently from the numbers used in AMFE and ParaView
                # (last two indices permuted)
                if shape == 'Tet10':
                    connectivity[np.array([9, 8], dtype=int)] = \
                        connectivity[np.array([8, 9], dtype=int)]
                # Same node numbering issue with Hexa20
                if shape == 'Hexa20':
                    hexa20_gmsh_swap = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 11,
                                                13, 9, 16, 18, 19,
                                                17, 10, 12, 14, 15], dtype=int)
                    connectivity[:] = connectivity[hexa20_gmsh_swap]

                builder.build_element(ele_id, shape, connectivity.tolist())

            index = index + n_ele_in_current_block + 1
            if entity_dimension in dim2entity2eleid:
                dim2entity2eleid[entity_dimension].update(
                    {entity_id: e_list.tolist()})
            else:
                dim2entity2eleid.update(
                    {entity_dimension: {entity_id: e_list.tolist()}})

        # Step 5:
        # Build Tags

        # Build elemental entity tags
        dim2elements = dict()
        entity2elements = dict()
        for dim in dim2entity2eleid.keys():
            current_arr = np.empty(shape=0, dtype=int)
            for entity_id, value in dim2entity2eleid[dim].items():
                current_arr = np.union1d(current_arr, np.array(value,
                                                               dtype=int))
                if entity_id in entity2elements:
                    entity2elements[entity_id] = np.union1d(
                        entity2elements[entity_id], np.array(value, dtype=int))
                else:
                    entity2elements[entity_id] = np.array(value, dtype=int)
            dim2elements[dim] = current_arr.tolist()
        for key, value in entity2elements.items():
            entity2elements[key] = value.tolist()
        builder.build_tag('gmsh_entity_dimension', dim2elements, int, 0)
        builder.build_tag('gmsh_entity_id', entity2elements, int, 0)

        # Build elemental partition tags
        partition_id2elements = dict()

        if has_partitions:
            for dim in dim2entities2partitions.keys():
                for ent_id in dim2entities2partitions[dim].keys():
                    partition_ids = tuple(dim2entities2partitions[dim][ent_id])
                    element_ids = dim2entity2eleid[dim][ent_id]
                    if partition_ids in partition_id2elements:
                        partition_id2elements[partition_ids] = np.union1d(
                            partition_id2elements[partition_ids],
                            np.array(element_ids, dtype=int)
                        )
                    else:
                        partition_id2elements[partition_ids] = np.array(
                            element_ids, dtype=int)
            for key, value in partition_id2elements.items():
                partition_id2elements[key] = value.tolist()
            builder.build_tag('gmsh_partition_ids', partition_id2elements,
                              default=())

        # Step 6:
        # Build groups
        groupname2dimphytag = dict()
        for dim in dim2phytag2groupname.keys():
            for tag, name in dim2phytag2groupname[dim].items():
                if name in groupname2dimphytag:
                    groupname2dimphytag[name].append((dim, tag))
                else:
                    groupname2dimphytag.update({name: [(dim, tag)]})

        for name, dim_and_tags in groupname2dimphytag.items():
            nodeids = np.empty(0, dtype=int)
            elementids = np.empty(0, dtype=int)
            for (dim, tag) in dim_and_tags:
                if dim == 0:
                    nodeids = np.union1d(nodeids, entity2nodeid[tag])
                entities = dim2phytag2entity[dim][tag]
                for entity in entities:
                    elementids = np.union1d(elementids,
                                            dim2entity2eleid[dim][entity])

            builder.build_group(name, nodeids.tolist(), elementids.tolist())

        # Other:
        builder.build_mesh_dimension(self._dimension)
        return
