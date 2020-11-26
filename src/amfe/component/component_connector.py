#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
from amfe.constraint.constraint_manager import *


class ComponentConnector:
    """
    The connector-class is a manager for all constraints and connections between components and has the according
    constraint-matrices. Constraint-matrices are stored in the
    constraints-dictionary with a pair of keys for each interface of the form '[slave-id]to[master-id]' and vice versa.
    """
    def __init__(self):
        self.constraints = dict()
        self._L = np.array([])
        self.dofs_mapping_local2global = dict()
        self.mesh_tying = MeshTying()

    def map_global_dofs2local(self, component_id, glo_dofidx):
        loc2glo_dofs_map = self.dofs_mapping_local2global[component_id]
        glo2loc_dofs_map = {glodof: locdof for locdof, glodof in loc2glo_dofs_map.items()}
        local_dofs = np.array([])

        for glodof_input in glo_dofidx:
            if glodof_input in glo2loc_dofs_map:
                local_dofs = np.append(local_dofs, glo2loc_dofs_map[glodof_input])

        return local_dofs

    def apply_compatibility_constraint(self, loc_master_id, master_component, loc_slave_id, slave_component):
        """
        Method to check compatibility of mesh-components and construct compatibility-constraints. It distinguishes
        between Master- and Slave-side of the interface.
        This determines currently the sign of the constraint-matrix only. The given local ids determine the
        corresponding keys in the constraints.
        The constraints are assumed to be enforced by Lagrange-multipliers.

        Parameters
        ----------
        loc_master_id : int
            local id of the component in the composite, which is on the interface's Master-side

        master_component : MeshComponent
            component of the Master-side, which has a mesh and a mapping

        loc_slave_id : int
            local id of the component in the composite, which is on the interface's Slave-side

        slave_component : MeshComponent
            component of the Slave-side, which has a mesh and a mapping

        Returns
        -------
        None
        """
        slave_nodeids, master_nodeids = self.mesh_tying.check_node_compatibility(slave_component.mesh,
                                                                                 master_component.mesh)
        
        master_key = (loc_slave_id, loc_master_id)
        slave_key = (loc_master_id, loc_slave_id)
        if slave_nodeids.size != 0 and master_nodeids.size != 0:
            slave_dofids = np.reshape(slave_component.mapping.get_dofs_by_nodeids(slave_nodeids), -1)
            master_dofids = np.reshape(master_component.mapping.get_dofs_by_nodeids(master_nodeids), -1)
            
            slave_constraints = ConstraintManager(slave_component.mapping.no_of_dofs)
            constraint = slave_constraints.create_dirichlet_constraint()
            name = 'Compatibility' + str(loc_master_id) + str(loc_slave_id)
            for dof in slave_dofids:
                slave_constraints.add_constraint(name, constraint, np.array([dof], dtype=int))
            X = slave_component.X
            u = np.zeros(X.shape)

            self.constraints[slave_key] = -slave_constraints.B(X, u, 0)
            
            master_constraints = ConstraintManager(master_component.mapping.no_of_dofs)
            constraint = master_constraints.create_dirichlet_constraint()
            name = 'Compatibility' + str(loc_slave_id) + str(loc_master_id)
            for dof in master_dofids:
                master_constraints.add_constraint(name, constraint, np.array([dof], dtype=int))
            X = master_component.X
            u = np.zeros(X.shape)

            self.constraints[master_key] = master_constraints.B(X, u, 0)

    def delete_connection(self, key):
        """
        Method to delete a connection by a given key.

        Parameters
        ----------
        key : string
            key of the constraint, which shall be removed

        Returns
        -------
        None
        """
        del self.constraints[key]

    def _assemble_constraint_matrices(self, component_ids, component_n_dofs):
        constraints_assembled = np.array([]).reshape(0, np.sum(component_n_dofs))
        used_keys = []

        def _assemble_matrices_rowwise_and_fill_with_zeros(left_idx, left_key, right_idx, right_key):
            start_zeros = np.array([]).reshape(n_rows, 0)
            middle_zeros = np.array([]).reshape(n_rows, 0)
            end_zeros = np.array([]).reshape(n_rows, 0)

            if int(left_key[0]) != component_ids[0]:
                n_fill_dofs = np.sum(component_n_dofs[0: left_idx])
                start_zeros = np.zeros((n_rows, n_fill_dofs))
            if int(right_key[0]) != component_ids[left_idx + 1]:
                n_fill_dofs = np.sum(component_n_dofs[left_idx + 1: right_idx])
                middle_zeros = np.zeros((n_rows, n_fill_dofs))
            if int(right_key[0]) != component_ids[-1]:
                n_fill_dofs = np.sum(component_n_dofs[right_idx + 1: len(component_n_dofs)])
                end_zeros = np.zeros((n_rows, n_fill_dofs))

            return np.hstack((start_zeros, self.constraints[left_key].todense(),
                              middle_zeros, self.constraints[right_key].todense(), end_zeros))

        for idx, compid in enumerate(component_ids):
            for key in self.constraints:
                if key[-1] == compid and key not in used_keys:
                    key_neighbor = (key[-1], key[0])

                    neighbor_idx = int(np.where(component_ids == int(key_neighbor[-1]))[0])

                    n_rows = int(self.constraints[key].shape[0])

                    if neighbor_idx > idx:
                        constraint_row_assembled = _assemble_matrices_rowwise_and_fill_with_zeros(idx, key,
                                                                                                  neighbor_idx,
                                                                                                  key_neighbor)
                    else:
                        constraint_row_assembled = _assemble_matrices_rowwise_and_fill_with_zeros(neighbor_idx,
                                                                                                  key_neighbor, idx,
                                                                                                  key)

                    constraints_assembled = np.vstack((constraints_assembled, constraint_row_assembled))

                    used_keys.append(key)
                    used_keys.append(key_neighbor)

        return constraints_assembled


class MeshTying:
    """
    Class, that checks relations between the meshes of different components.
    """
    def __init__(self):
        pass

    @staticmethod
    def check_node_compatibility(mesh_a, mesh_b, tol=1e-12):
        """
        Method, that checks, if nodes of two components with meshes are approximately at the same position. This is only
        based on the reference-coordinates. So no check during a simulation is yet supported.

        Parameters
        ----------
        mesh_a : Mesh
            first mesh
        mesh_b : Mesh
            second mesh
        tol: float
            tolerated distance between nodes that are assumed to match

        Returns
        -------
        nodeids_a : ndarray
            node-ids of component_a, that match nodeids_b of component_b

        nodeids_b : ndarray
            node-ids of component_b, that match nodeids_a of component_a
        """
        nodeids_a_full = mesh_a.nodes_df.index.values
        nodeids_a = np.array([], dtype=int)
        nodeids_b = np.array([], dtype=int)

        for node_a in nodeids_a_full:
            node_a_coord = mesh_a.nodes_df.loc[node_a]
            if mesh_a.dimension == 2:
                matching_nodeid_b = mesh_b.get_nodeid_by_coordinates(node_a_coord['x'], node_a_coord['y'],
                                                                     epsilon=tol)
            elif mesh_a.dimension == 3:
                matching_nodeid_b = mesh_b.get_nodeid_by_coordinates(node_a_coord['x'], node_a_coord['y'],
                                                                     node_a_coord['z'], epsilon=tol)
            else:
                raise ValueError('Mesh must be dimension 2 or 3')
            
            if matching_nodeid_b is not None:
                nodeids_b = np.append(nodeids_b, matching_nodeid_b)
                nodeids_a = np.append(nodeids_a, node_a)
                
        return nodeids_a, nodeids_b
