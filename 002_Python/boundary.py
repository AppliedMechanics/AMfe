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



class NeumannBoundary():
    '''Class for application of von Neumann coundary conditions. Works a little bit crazy but it's working.
    '''

    def __init__(self, ndof_global, neumann_boundary_list):
        self.neumann_boundary_list = neumann_boundary_list
        self.ndof_global = ndof_global
        pass

    def _harmonic_excitation(self, amplitude, frequency):
        '''
        function returning the harmonic function
        '''

        def internal_harmonic(t):
            return amplitude*np.cos(frequency*2*np.pi*t)

        return internal_harmonic


    def _step_function(self, step_size, time):
        '''
        function returning a step function
        '''
        def internal_step_function(t):
            if t > time:
                return step_size
            else:
                return 0

        return internal_step_function

    def _dirac_impulse(self, amplitude, time, time_interval):
        '''
        returns a dirac impulse as excitation
        '''

        def internal_dirac(t):
            if abs(t - time) < time_interval:
                return amplitude / time_interval
            else:
                return 0

        return internal_dirac

    def _ramp_function(self, slope, time):
        '''returns a ramp function '''

        def internal_ramp(t):
            delta_t = t - time
            if delta_t < 0:
                return 0
            else:
                return delta_t*slope

        return internal_ramp

    def _constant_function(self, amplitude):
        '''returns a constant function; this makes most sense for static applications'''

        def internal_constant(t):
            return amplitude

        return internal_constant

    function_dict = {'harmonic': _harmonic_excitation,
                     'stepload': _step_function,
                     'dirac'   : _dirac_impulse,
                     'ramp'    : _ramp_function,
                     'static'  : _constant_function}

    def f_ext(self):
        '''
        Input:
        ------
        no input is required for this function

        Output:
        -------
        a function f, which can be called with f(t) giving back the von neumann bcs

        '''
        self.function_list = []
        counter = 0
        row_global = np.array([], dtype=int)
        col_global = np.array([], dtype=int)
        vals_global = np.array([], dtype=float)

        for dofs, type_, props, B_matrix in self.neumann_boundary_list:
            # constructing the indices for the boolean matrix grouping the functions in the right place
            col = np.ones(len(dofs), dtype=int)*counter
            row = np.array(dofs)
            if B_matrix:
                vals = B_matrix
            else:
                vals = np.ones(len(dofs))

            row_global  = np.append(row_global, row)
            col_global  = np.append(col_global, col)
            vals_global = np.append(vals_global, vals)
            counter += 1

            # construct_the_list
            self.function_list.append(self.function_dict[type_](self, *props))

        self.boolean_force_matrix = sp.sparse.csr_matrix((vals, (row_global, col_global)), shape=(self.ndof_global, len(self.function_list)))

        # export external forcing function
        def external_forcing_function(t):
            return self.boolean_force_matrix.dot(np.array([i(t) for i in self.function_list]))
        return external_forcing_function


# TEST
if __name__ == '__main__':
    my_neumann_bc = NeumannBoundary(2000, [[[0,], 'dirac', (5, 0.5, 1E-3), None],])
    f = my_neumann_bc.f_ext()
    T = np.arange(-1, 5, 0.1)
    res = np.array([f(t) for t in T])
    from matplotlib import pyplot
    pyplot.plot(T, res[:, 0])

