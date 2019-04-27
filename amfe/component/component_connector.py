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
        self.mesh_tying = MeshTying()
    
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
        
        master_key = str(loc_slave_id) + 'to' + str(loc_master_id)
        slave_key = str(loc_master_id) + 'to' + str(loc_slave_id)
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

            self.constraints[slave_key] = -slave_constraints.B(X, u , 0)
            
            master_constraints = ConstraintManager(master_component.mapping.no_of_dofs)
            constraint = master_constraints.create_dirichlet_constraint()
            name = 'Compatibility' + str(loc_slave_id) + str(loc_master_id)
            for dof in master_dofids:
                master_constraints.add_constraint(name, constraint, np.array([dof], dtype=int))
            X = master_component.X
            u = np.zeros(X.shape)
            
            self.constraints[master_key] = master_constraints.B(X, u , 0)
            
        elif master_key in self.constraints or slave_key in self.constraints:
            self.delete_connection(master_key)
            self.delete_connection(slave_key)
            
    def delete_connection(self, key):
        """
        Method to delete a connection by a given key. Returns an error if the key is not available in the constraints.

        Parameters
        ----------
        key : string
            key of the constraint, which shall be removed

        Returns
        -------
        None
        """
        try:
            del self.constraints[key]
        except KeyError:
            pass
    

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
