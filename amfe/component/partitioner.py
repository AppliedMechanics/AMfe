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
        dof_mapping_loc2glo_list = []
        materials = component.get_materials()
        physics = component.get_physics()
        fields = component.fields
        for partition_id in mesh.get_uniques_by_tag('partition_id'):
            submesh = self._get_submesh_by_partition_id(partition_id, mesh)
            
            new_component = component.__class__(submesh)
            self._assign_materials_to_new_component(new_component, component, materials, physics)
            partitions_nodes = nodes_mapping[nodes_mapping['partition_id'] == partition_id]
            dof_mapping_loc2glo = self._map_dofs_local2global(partitions_nodes, new_component._mapping,
                                                              component._mapping, fields)

            new_components_list.append(new_component)
            dof_mapping_loc2glo_list.append(dof_mapping_loc2glo)
        return new_components_list, dof_mapping_loc2glo_list

    @staticmethod
    def _map_dofs_local2global(nodes_mapping_df, local_component_mapping, global_component_mapping, fields):
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
        
        ele_ids = mesh.get_elementids_by_tags('partition_id', partition_id)
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
            ele_ids = mesh.get_elementids_by_tags('partition_id', partition_id)

            nodeids_full = mesh.get_nodeids_by_elementids(ele_ids)

            for node in nodeids_full:
                if node in nodes_mapping['global_nodeid'].values:
                    new_node = mesh.copy_node_by_id(node)
                    nodes_mapping = nodes_mapping.append(
                        {'partition_id': partition_id, 'global_nodeid': node, 'local_nodeid': new_node}, ignore_index=True)
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
                assign_eleids = np.intersect1d(common_full_ele, new_component._mesh._el_df.index)
                new_component.assign_material(material, assign_eleids, phys, '_eleids')
