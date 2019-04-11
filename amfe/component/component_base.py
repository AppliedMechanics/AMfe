# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#


class ComponentBase:
    
    def __init__(self, *args, **kwargs):
        pass
    
    def get_mat(self, matrix_type, q, dq, t):
        """
        Returns a requested matrix dependend on given path

        Parameters
        ----------
        matrix_type : str
            Matrix type that is returned (e.g. M, K, ...)
        u : ndarray
            primal variable (e.g. displacements)
        t : float
            time

        Returns
        -------
        matrix : ndarray or csc_matrix
            the requested matrix
        """
        func = getattr(self, matrix_type)
        return func(q, dq, t)
    
    def unconstrain_vector(self, vector):
        return vector

    def partition(self, no_of_components, element_id_sets):
        """
        TODO: Implement subroutines to split up component
        """
        pass
