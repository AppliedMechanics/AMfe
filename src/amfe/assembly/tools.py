#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Tools for assembly module.
"""

__all__ = [
    'get_index_of_csr_data',
    'fill_csr_matrix'
]


# try to import the fortran routines
use_fortran = False
try:
    import amfe.f90_assembly
    use_fortran = True
except ImportError:
    print('Python was not able to load the fast fortran assembly routines.')


def get_index_of_csr_data(i, j, indptr, indices):
    """
    Get the value index of the i,j-element of a matrix in CSR format.

    Parameters
    ----------
    i : int
        row index which is asked to get the CSR-index for
    j : int
        column index which is asked to get the CSR-index for
    indptr : ndarray
        index-ptr-Array of the CSR-Matrix.
    indices : ndarray
        indices array of CSR-matrix (represents the nonzero column indices)

    Returns
    -------
    k : int
        index of the value array of the CSR-matrix, in which value [i,j] is stored.

    Notes
    -----
    This routine works only, if the tuple i,j is acutally a real entry of the matrix. Otherwise the value k=0 will be
    returned and an Error Message will be provided.
    """

    # indices for row i are stored in indices[indptr[k]:indptr[k+1]]; thus the indptr marks the start and end of the
    # part of the indices and val vector where all entries of a row are stored

    # set k to the start of data of row k
    k = indptr[i]
    # search for appearance of j in the nonzero column indices which are stored in indices[k] till
    # indices[k+indptr[i+1]]
    while j != indices[k]:
        # while column j not found search for j in next entry
        k += 1
        # Check if next search would be in next (wrong) row
        if k > indptr[i + 1]:
            print('ERROR! The index in the csr matrix is not preallocated!')
            k = 0
            break
    return k


def fill_csr_matrix(indptr, indices, vals, K, k_indices):
    """
    Fill the values of K into the vals-array of a sparse CSR Matrix given the k_indices array. The values of K are
    added to the current values (typically for assembly processes)

    Parameters
    ----------
    indptr : ndarray
        indptr-array of a preallocated CSR-Matrix
    indices : ndarray
        indices-array of a preallocated CSR-Matrix
    vals : ndarray
        vals-array of a preallocated CSR-Marix
    K : ndarray
        'small' square array whose values will be distributed into the
        CSR-Matrix, Shape is (n,n)
    k_indices : ndarray
        mapping array of the global indices for the 'small' K array.
        The (i,j) entry of K has the global indices (k_indices[i], k_indices[j])
        Shape is (n,)

    Returns
    -------
    None

    """
    ndof_l = K.shape[0]
    for i in range(ndof_l):
        for j in range(ndof_l):
            l = get_index_of_csr_data(k_indices[i], k_indices[j], indptr, indices)
            vals[l] += K[i, j]
    return


if use_fortran:
    ###########################################################################
    # Fortran routine that will override the functions above for massive speedup.
    ###########################################################################
    get_index_of_csr_data = amfe.f90_assembly.get_index_of_csr_data
    fill_csr_matrix = amfe.f90_assembly.fill_csr_matrix
