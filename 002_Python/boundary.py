# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:28:07 2015

@author: johannesr
"""

import numpy as np
import scipy as sp



class DirichletBoundary():
    '''
    Class responsible for the Dirichlet Boundary conditions

    The boundary-information is stored in the master_slave_list, which forms the interface for all homogeneous Dirichlet boundary condtions.

    The Master-Slave-List is organized as follows:

    '''
    def __init__(self, ndof_full_system, master_slave_list=None):
        self.ndof_full_system = ndof_full_system
        self.master_slave_list = master_slave_list
        pass


    def b_matrix(self):
        '''
        Parameters
        ----------
        no parameters

        Information
        -----------


        Erzeugt die B-Matrix, die die globalen (u_global) Freiheitsgrade mit den beschränkten Freiheitsgraden (u_bound) verbindet:

        u_global = B*u_bound

        Die globalen Freiheitsgrade differieren daher von den Freiheitsgraden des beschränkten Systems;
        Eingabeparameter ist eine Liste mit Dirichlet-Randbedingungen:
        [Master-DOF, [Liste_von_Sklaven-DOFs], Gewichtungsvektor]

        Master-DOF: (typ: int) Der DOF, auf den die Sklaven-DOFs projiziert werden. Der Master-DOF wird am ende eliminiert, d.h. er sollte üblicherweise auch in den Sklaven-DOFs auftauchen

        [Liste_von_Sklaven-DOFs]: (typ: liste mit ints) Die DOFs, die auf den Master-DOF projiziert werden. Zur Gewichtung wird der Gewichtungsvektor angewendet, der genauso viele Einträge haben muss wie die Sklaven-DOF-Liste

        Gewichtungsvektor: (typ: np.array oder None) TODO Beschreibung


        Wichtig: Für die Dirichlet-Randbedingungen werden Freiheitsgrade des globalen Systems und nicht die Knotenfreiheitsgrade berücksichtigt. Die Indexwerte der Knoten müssen stets in DOFs des globalen Sytems umgerechnet werden
        '''
        B = sp.sparse.eye(self.ndof_full_system).tocsr()
        B_tmp = B*0
        global_slave_node_list = np.array([], dtype=int)
        global_master_node_list = np.array([], dtype=int)

        # Loop over all boundary items; the boundary information is stored in the _tmp-Variables
        for master_node, slave_node_list, b_matrix in self.master_slave_list:

            if (type(b_matrix) != type(np.zeros(1))): # a little hack in order to get the types right
                # Make a B-Matrix, if it's not there
                b_matrix = np.ones(len(slave_node_list))

            if len(slave_node_list) != len(b_matrix):
                raise ValueError('Die Dimension der Sklaven-Knotenliste entspricht nicht der Dimension des Gewichtungsvektors!')

            if master_node != None: # check, if the master node is existent; otherwise the columns will only be deleted
                for i in range(len(slave_node_list)):
                    ## This is incredible slow!!!, but it's exactly what is done:
                    # B_tmp[:,master_node] += B[:,i]*b_matrix[i]
                    # so here's the alternative:
                    col = B[:,slave_node_list[i]]*b_matrix[i]
                    row_indices = col.nonzero()[0]
                    no_of_nonzero_entries = row_indices.shape[0]
                    col_indices = np.ones(no_of_nonzero_entries)*master_node
                    B_tmp = B_tmp + sp.sparse.csr_matrix((col.data, (row_indices, col_indices)), shape=(self.ndof_full_system, self.ndof_full_system))

                global_master_node_list = np.append(global_master_node_list, master_node)
            global_slave_node_list = np.append(global_slave_node_list, slave_node_list)

        # Remove the master-nodes from B as they have to be overwritten by the slaves
        for i in global_master_node_list:
            B[i, i] -= 1
        B = B + B_tmp

        # Remove the master-nodes from the slave_node_list and mast the matrix such, that the slave_nodes are removed
        global_slave_node_list = np.array([i for i in global_slave_node_list if i not in global_master_node_list])
        mask = np.ones(self.ndof_full_system, dtype=bool)
        mask[global_slave_node_list] = False
        B = B[:,mask]
        return B

