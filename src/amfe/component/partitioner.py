#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
import numpy as np
import pandas as pd
from copy import deepcopy
from amfe.mesh import Mesh
from amfe.tools import invert_dictionary_with_iterables


class PartitionerBase:
    def __init__(self):
        pass
    
    def partition(self):
        raise NotImplementedError('Partitioning is not implemented for subclass!')


class MetisPartitioner(PartitionerBase):
    def __init__(self):
        super().__init__()
        pass
        
    def partition(self):
        pass


class PartitionedComponentSeparator:
    def __init__(self):
        pass
        
    def separate_partitioned_component(self, component):
        pass


class PartitionedMeshComponentSeparator(PartitionedComponentSeparator):
    """
    After running a partitioning algorithm on a component's mesh, that component might have a mesh with elements, which
    are marked by partition-ids. Hence this mesh still has to be subdivided into submeshes according to the partitions.
    Furthermore the component has to be subdivided into components according to the new meshes, as a component may only
    have one unique mesh.

    
    That's what this class does. It provides methods to separate a component with a partitioned mesh into further
    components of the same type.

    """
    def __init__(self):
        super().__init__()

    def set_partition_tags_by_group(self, mesh, group_subname='surface'):
        """
        Resets partition tags in a mesh such that each combination of previous partition-id and tag-type-value is
        unique. Hence for example no partition is on two different surfaces any more, as it might result from the
        partitioning algorithm.

        Parameters
        ----------
        mesh : Mesh
            mesh-object with partitions
        group_subname : str
            keyword, that shall be searched in the group-names

        Returns
        -------
        None
        """
        elements_by_tags = self._select_element_subset_by_group(mesh, group_subname)

        elements_by_partition_ids = self._get_element_subset_by_partition_ids(mesh)

        partition_map = self._create_new_partition_map(mesh, elements_by_tags, elements_by_partition_ids)

        mesh.el_df = self._update_elements_with_new_partition_ids(mesh.el_df, partition_map, mesh.nodes_df)

    def separate_partitioned_component(self, component):
        """
        Seperates a component, which has a mesh, into several components of the same type. Hence its mesh has to be
        partitioned already, either by importation or by a partitioning-algorithm.

        
        Parameters
        ----------
        component : MeshComponent

        Returns
        -------
        new_components_list : list of MeshComponent, StructuralComponent
        """
        mesh, nodes_mapping = self._separate_common_nodes_of_partitions(component._mesh)
        
        new_components_list = []
        new_ids_list = []
        dof_mapping_loc2glo_dict = dict()
        materials = component.get_materials()
        physics = component.get_physics()
        fields = component.fields
        for partition_id in mesh.get_uniques_by_tag('partition_id'):
            submesh = self._get_submesh_by_partition_id(partition_id, mesh)
            
            new_component = component.__class__(submesh)
            self._assign_materials_to_new_component(new_component, component, materials, physics)
            partitions_nodes = nodes_mapping[nodes_mapping['partition_id'] == partition_id]
            dof_mapping_loc2glo = self._map_dofs_local2global(partitions_nodes, new_component.mapping,
                                                              component.mapping, fields)

            new_ids_list.append(partition_id)
            new_components_list.append(new_component)
            dof_mapping_loc2glo_dict[partition_id] = dof_mapping_loc2glo
        return new_ids_list, new_components_list, dof_mapping_loc2glo_dict

    @staticmethod
    def _map_dofs_local2global(nodes_mapping_df, local_component_mapping, global_component_mapping, fields):
        """
        Generates a mapping of the new local dofs to the old global dofs of the unseparated component from a nodes-
        mapping and both nodes-to-dofs-mappings of the original component and the new local component.

        Parameters
        ----------
        nodes_mapping_df : pandas.DataFrame
            mapping of old global node-ids to the new local node-ids
        local_component_mapping : MappingBase
            nodes-to-dofs mapping of new local component
        global_component_mapping : MappingBase
            nodes-to-dofs mapping of old global component
        fields : tuple of strings
            all fields of the old global component

        Returns
        -------
        dof_mapping_loc2glo : dict
            mapping of local dofs to old global dofs
        """
        dof_mapping_loc2glo = dict()
        for idx, node_map in nodes_mapping_df.iterrows():
            glo_nodeid = node_map.loc['global_nodeid']
            loc_nodeid = node_map.loc['local_nodeid']

            for field in fields:
                loc_dofid = local_component_mapping.get_dofs_by_nodeids(loc_nodeid, field)
                glo_dofid = global_component_mapping.get_dofs_by_nodeids(glo_nodeid, field)

                dof_mapping_loc2glo[loc_dofid[0]] = glo_dofid[0]

        return dof_mapping_loc2glo

    @staticmethod
    def _get_submesh_by_partition_id(partition_id, mesh):
        """
        Getter for a submesh, derived from the given mesh, but only possesses elements, nodes and groups, which belong
        to the given partition.
        
        Parameters
        ----------
        partition_id : int
            ID of the partition
            
        mesh : Mesh
            partitioned Mesh-object
        """
        submesh = Mesh(mesh.dimension)
        
        ele_ids = mesh.get_elementids_by_tags(['partition_id'], partition_id)
        nodes_df, el_df = mesh.get_submesh_by_elementids(ele_ids)
        ele_groups = mesh.get_groups_dict_by_elementids(el_df.index.tolist())
        node_groups = mesh.get_groups_dict_by_nodeids(nodes_df.index.tolist())
        submesh.nodes_df = deepcopy(nodes_df)
        submesh._el_df = deepcopy(el_df)
        submesh.merge_into_groups(ele_groups)
        submesh.merge_into_groups(node_groups)
        submesh._update_iconnectivity()
        
        return submesh

    @staticmethod
    def _separate_common_nodes_of_partitions(mesh):
        """
        If a partitioned mesh is imported or an unpartitioned mesh is divided into partitions by a partitioning-method
        in AMfe, several partitions might share common nodes. As this prevents the unique distribution of submeshes to
        components and thus maybe CPU-cores, it is necessary to divide these common nodes and assign the new unique
        nodes to a unique submesh.
        
        This method searches such shared nodes, copies them and adjusts the elements' connectivities and groups. After
        separation a control-flag is set 'True'.
        
        Parameters
        ----------
        mesh : Mesh
        
        Returns
        -------
        mesh : Mesh

        copied_nodes : pd.DataFrame
        """
        mesh = deepcopy(mesh)
        nodes_mapping = pd.DataFrame(columns=('partition_id', 'global_nodeid', 'local_nodeid'))
        nodes_mapping = nodes_mapping.astype(int)

        for partition_id in mesh.get_uniques_by_tag('partition_id'):
            ele_ids = mesh.get_elementids_by_tags(['partition_id'], partition_id)

            nodeids_full = mesh.get_nodeids_by_elementids(ele_ids)

            for node in nodeids_full:
                if node in nodes_mapping['global_nodeid'].values:
                    new_node = mesh.copy_node_by_id(node)
                    nodes_mapping = nodes_mapping.append(
                        {'partition_id': partition_id, 'global_nodeid': node, 'local_nodeid': new_node},
                        ignore_index=True)
                    groups_node = mesh.get_groups_by_nodeids(node)
                    mesh.add_node_to_groups(new_node, groups_node)

                    mesh.update_connectivity_with_new_node(node, new_node, ele_ids)
                else:
                    nodes_mapping = nodes_mapping.append(
                        {'partition_id': partition_id, 'global_nodeid': node, 'local_nodeid': node}, ignore_index=True)

        return mesh, nodes_mapping

    @staticmethod
    def _assign_materials_to_new_component(new_component, old_component, old_materials, old_physics):
        for material in old_materials:
            full_mat_ele = old_component.get_elementids_by_materials(material)
            for phys in old_physics:
                full_phys_ele = old_component.get_elementids_by_physics(phys)
                common_full_ele = np.intersect1d(full_mat_ele, full_phys_ele)
                assign_eleids = np.intersect1d(common_full_ele, new_component.mesh.el_df.index)
                new_component.assign_material(material, assign_eleids, phys, '_eleids')

    @staticmethod
    def _select_element_subset_by_group(mesh, group_subname):
        subgroups = dict()

        group_names = tuple(mesh.groups.keys())
        for group in group_names:
            if group_subname in group:
                ele_ids = mesh.get_elementids_by_groups((group,))
                if len(ele_ids) is not 0:
                    subgroups[group] = ele_ids

        return subgroups

    @staticmethod
    def _get_element_subset_by_partition_ids(mesh):
        partitions_old_list = mesh.get_uniques_by_tag('partition_id')
        partition_id_2_eleids_old = dict()
        for part_id in partitions_old_list:
            partition_id_2_eleids_old[part_id] = mesh.get_elementids_by_tags('partition_id', part_id)

        return partition_id_2_eleids_old

    @staticmethod
    def _create_new_partition_map(mesh, element_set_1, element_set_2):
        partition_id = 1
        partition_map = dict()
        eles_that_have_partition_id = np.array([])

        for group_id, group_eles in element_set_1.items():
            for part_id, part_eles in element_set_2.items():
                new_eles = np.intersect1d(group_eles, part_eles)
                if new_eles.size is not 0:
                    partition_map[partition_id] = new_eles
                    eles_that_have_partition_id = np.append(eles_that_have_partition_id, new_eles)
                    partition_id += 1

        eles_without_partition = np.setdiff1d(mesh.el_df.index.values, eles_that_have_partition_id)
        if eles_without_partition.size is not 0:
            for eleid in eles_without_partition:
                nodeids = mesh.get_nodeids_by_elementids(eleid)
                match_found = False
                for part_id, part_eles in partition_map.items():
                    for part_ele in part_eles:
                        other_ele_nodeids = mesh.get_nodeids_by_elementids(part_ele)
                        if np.intersect1d(nodeids, other_ele_nodeids).size > 1:
                            partition_map[part_id] = np.append(partition_map[part_id], eleid)
                            match_found = True
                            break
                    if match_found:
                        break

        return partition_map

    @staticmethod
    def _update_elements_with_new_partition_ids(el_df, partition_map, nodes_df):
        ele2partition_dict = invert_dictionary_with_iterables(partition_map)
        ele2partition_map = pd.DataFrame.from_dict(ele2partition_dict, orient='index')
        el_df['partition_id'] = ele2partition_map

        def check_neighbor(element1, element2):
            is_neighbor = False
            if element1['partition_id'] is not element2['partition_id']:
                for node1 in element1['connectivity']:
                    for node2 in element2['connectivity']:
                        if np.isclose(nodes_df.loc[node1].to_numpy(), nodes_df.loc[node2].to_numpy()).all():
                            is_neighbor = True
                            break
                    if is_neighbor:
                        break
            return is_neighbor

        for eleid, element in el_df.iterrows():
            no_of_partitions = 1
            partitions_neighbors = tuple()
            for other_eleid, other_element in el_df.iterrows():
                if other_eleid is not eleid and other_element['partition_id'] not in partitions_neighbors and \
                        check_neighbor(element, other_element):
                    partitions_neighbors += (other_element['partition_id'],)
            no_of_partitions += len(partitions_neighbors)
            if len(partitions_neighbors) is 0:
                partitions_neighbors = None
            elif len(partitions_neighbors) is 1:
                partitions_neighbors = partitions_neighbors[0]
            else:
                partitions_neighbors = list(partitions_neighbors)
                partitions_neighbors.sort()
                partitions_neighbors = tuple(partitions_neighbors)
            el_df.at[eleid, 'partitions_neighbors'] = partitions_neighbors
            el_df.at[eleid, 'no_of_mesh_partitions'] = no_of_partitions

        return el_df
