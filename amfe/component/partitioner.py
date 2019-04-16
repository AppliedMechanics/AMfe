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
from collections.abc import Iterable
from numpy import partition


class PartitionerBase:
    def __init__(self):
        pass
    
    def partition(self):
        raise NotImplementedError('Partitioning is not implemented for subclass!')
    
class MetisPartitioner(PartitionerBase):
    def __init__(self):
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
    After running a partitioning algorithm on a component's mesh, that component might have a mesh with elements, which are marked by partition-ids. Hence this mesh still has to be subdivided
    into submeshes according to the partitions. Furthermore the component has to be subdivided into components according to the new meshes, as a component may only have one unique mesh. 
    
    That's what this class does. It provides methods to separate a component with a partitioned mesh into further components of the same type. 
    """
    def __init__(self):
        super().__init__()
    
    def separate_partitioned_component(self, component, opt_keep_node_ids=False):
        """
        Seperates a component, which has a mesh, into several components of the same type. Hence its mesh has to be partitioned already, either by importation or by a partitioning-algorithm.
        
        Parameters
        ----------
        component : MeshComponent
        
        opt_keep_node_ids : Boolean
            Option to keep node-ids and skip the duplication routines. Therefore components will have nodes with the same ids in their meshes.
        
        Returns
        -------
        new_components_list : list of MeshComponent, StructuralComponent
        """
        if opt_keep_node_ids:
            mesh = component._mesh
        else:
            mesh = self._separate_common_nodes_of_partitions(component._mesh)
        
        new_components_list = []
        materials = component.get_materials()
        physics = component.get_physics()
        for partition_id in mesh.get_uniques_by_tag('partition_id'):
            submesh = self._get_submesh_by_partition_id(partition_id, mesh)
            
            new_component = component.__class__(submesh)
            for material in materials:
                full_mat_ele = component.get_elementids_by_materials(material)
                for phys in physics:
                    full_phys_ele = component.get_elementids_by_physics(phys)
                    common_full_ele = np.intersect1d(full_mat_ele, full_phys_ele)
                    assign_eleids = np.intersect1d(common_full_ele, submesh._el_df.index.values)
                    
                    new_component.assign_material(material, assign_eleids, phys, '_eleids')

            new_components_list.append(new_component)
            
        return new_components_list
    
    def _get_submesh_by_partition_id(self, partition_id, mesh):
        """
        Getter for a submesh, derived from the given mesh, but only possesses elements, nodes and groups, which belong to the given partition.
        
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
    
    def _separate_common_nodes_of_partitions(self, mesh):
        """
        If a partitioned mesh is imported or an unpartitioned mesh is divided into partitions by a partitioning-method in AMfe, several partitions might share common nodes.
        As this prevents the unique distribution of submeshes to components and thus maybe CPU-cores, it is necessary to divide these common nodes and assign the new unique nodes to a unique submesh.
        
        This method searches such shared nodes, copies them and adjusts the elements' connectivities and groups. After separation a control-flag is set 'True'.
        
        Parameters
        ----------
        mesh : Mesh
        
        Returns
        -------
        mesh : Mesh
        """
        mesh = deepcopy(mesh)
        copied_nodes = pd.DataFrame(columns=('partition_id', 'old_node', 'new_node'))

        for partition_id in mesh.get_uniques_by_tag('partition_id'):
            ele_ids = mesh.get_elementids_by_tags(['no_of_mesh_partitions', 'partition_id'], [1 ,partition_id], [True, False])
            for ele_id in ele_ids:
                neighbor_part_ids = mesh.get_value_by_elementid_and_tag(ele_id, 'partitions_neighbors')
                if not isinstance(neighbor_part_ids, Iterable):
                    neighbor_part_ids = [neighbor_part_ids]
                for inode in mesh.get_nodeids_by_elementids(ele_id):
                    for i_n_part in neighbor_part_ids:
                        neighbor_eleids = mesh.get_elementids_by_tags(['no_of_mesh_partitions', 'partition_id'], [1 ,i_n_part], [True, False])
                        neighbor_nodes = mesh.get_nodeids_by_elementids(neighbor_eleids)
                        if inode in neighbor_nodes: 
                            if inode not in copied_nodes[copied_nodes['partition_id'] == i_n_part]['old_node'].values:
                                new_node = mesh.copy_node_by_id(inode)
                                
                                copied_nodes = copied_nodes.append({'partition_id': i_n_part, 'old_node': inode, 'new_node': new_node}, ignore_index=True)

                                groups_node = mesh.get_groups_by_nodeids(inode)
                                mesh.add_node_to_groups(new_node, groups_node)
      
                            mesh.update_connectivity_with_new_node(inode, new_node, neighbor_eleids)
        
        return mesh
    
    
    