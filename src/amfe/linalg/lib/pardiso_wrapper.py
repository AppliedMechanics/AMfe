# Copyright (c) 2016 David Marchant
#
# Originally distributed under MIT License
#
# Modified by Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# and
# Redistributed under BSD-3-Clause License. See LICENSE-File for more information
#
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
# from future import standard_library
# standard_library.install_aliases()
from builtins import object


from ctypes import POINTER, byref, c_longlong, c_int
import numpy as np
import scipy.sparse as sp
from numpy import ctypeslib

"""
mtype options
  1 -> real and structurally symmetric
  2 -> real and symmetric positive definite
 -2 -> real and symmetric indefinite
  3 -> complex and structurally symmetric
  4 -> complex and Hermitian positive definite
 -4 -> complex and Hermitian indefinite
  6 -> complex and symmetric
 11 -> real and nonsymmetric
 13 -> complex and nonsymmetric


phase options
 11 -> Analysis
 12 -> Analysis, numerical factorization
 13 -> Analysis, numerical factorization, solve, iterative refinement
 22 -> Numerical factorization
 23 -> Numerical factorization, solve, iterative refinement
 33 -> Solve, iterative refinement
331 -> like phase=33, but only forward substitution
332 -> like phase=33, but only diagonal substitution (if available)
333 -> like phase=33, but only backward substitution
  0 -> Release internal memory for L and U matrix number mnum
 -1 -> Release all internal memory for all matrices
"""

try:
    from .pardisoInterface import pardisoinit, pardiso

    class PardisoWrapper(object):
        """Wrapper class for Intel MKL Pardiso solver. """
        def __init__(self, A, mtype=11, iparm=None, verbose=False):
            """
            Parameters
            ----------
            A : scipy.sparse.csr.csr_matrix
                sparse matrix in csr format.
            mtype : int, optional
                flag specifying the matrix type. The possible types are:

                - 1 : real and structurally symmetric (not supported)
                - 2 : real and symmetric positive definite
                - -2 : real and symmetric indefinite
                - 3 : complex and structurally symmetric (not supported)
                - 4 : complex and Hermitian positive definite
                - -4 : complex and Hermitian indefinite
                - 6 : complex and symmetric
                - 11 : real and nonsymmetric (default)
                - 13 : complex and nonsymmetric
            verbose : bool, optional
                flag for verbose output. Default is False.

            iparm: dictionary with iparm values to override after initialization with mtype
                iparm[0] : {0: fill rest iparms with default values, 1: use specified values}
                iparm[1] : {0: The minimum degree algorithm, 2: Nested dissection (from METIS) (default),
                                3: Parallel (OpenMP version) of nested dissection}
                iparm[2] : {0: Reserved, set to zero (default)}
                iparm[3] : Preconditioned CGS/CG: This parameter controls preconditioned CGS for nonsymmetric or
                           structurally symmetric matrices and Conjugate-Gradients for symmetric matrices.
                           See https://software.intel.com/en-us/mkl-developer-reference-fortran-pardiso-iparm-parameter
                           for details
                iparm[4] : {0: User permutation in the perm array is ignored (default),
                            1: use user supplied fill-in reducing permutation from the perm array (iparm[1] is ignored)
                            2: Return the permutation vector computed at phase 1 in the perm array}
                iparm[5] : {0: The array x contains the solution (default),
                            1: The solver stores the solution on the RHS b}
                iparm[6] : (Output parameter): The number of iterative refinement steps performed
                iparm[7] : Iterative Refinement step (maximum number) {0: The solver automatically performs 2 steps when
                            perturbed pivots are obtained during factorization,
                            >0: maximum number of iterative refinement steps,
                            <0 same as >0 but the accumulation of residuum uses extended precision data types}
                iparm[8] : Reserved set to zero
                iparm[9] : Pivoting perturbation, small pivots are perturbed with eps = 10**(-iparm[9])
                           default value for nonsymmetric matrices: 13,
                           default value for symmetric indefinite matrices: 8
                iparm[10] : Permutation and scaling: {0: scaling diabled (default for symmetric indefinite matrices),
                            1: Enable scaling (default for nonsymmetric matrices)
                            Note: Use iparm[10] = 1 and iparm[12] = 1 for saddle point problems
                iparm[11] : {0: Solve Ax = b (default), 1: Solve A^H = b (hermite), 2: Solve A^T = b (transposed)}
                iparm[12] : Use symmetric weighted matching {0: Disable matching (default for symmetric indefinite mat.)
                            , 1: Enable matching (default for nonsymmetric matrices)
                            Note: Use iparm[12] = 1 and iparm[10] = 1 for saddle point problems
                iparm[13] : Output: Number of perturbed pivots
                iparm[14] : Output: Peak memory on symbolic factorization in kilobytes
                iparm[15] : Output: Permanent memory on symbolic factorization
                iparm[16] : Output: Size of factors/peak memory on numerical factorization and solution in kB
                            The total peak memory is: max(iparm[14], iparm[15]+ iparm[16])
                iparm[17] : input/output: non-zero elements in the factors
                            {<0: Enable reporting (default: -1), >0 Disable reporting}
                iparm[18] : input/output: Report number of floating point operations in 10^6 floating point operations
                            that are necessary to factorize A {<0: Enable Report (this increases reordering time),
                            >0: disable report
                iparm[19] : Output: Report CG/CGS diagnostics {>0: CG/CGS succeeded (no. of completed iterations),
                            <0: failed: see
                            https://software.intel.com/en-us/mkl-developer-reference-fortran-pardiso-iparm-parameter for
                            more details of this number}
                iparm[20] : Pivorting for symmetric indefinite matrices
                            see https://software.intel.com/en-us/mkl-developer-reference-fortran-pardiso-iparm-parameter
                            for details, default: 1: Apply 1x1 and 2x2 Bunch-Kaufmann pivoting
                iparm[21] : Output: Inertia: number of positive eigenvalues
                iparm[22] : Output: Inertia: number of negative eigenvalues
                iparm[23] : Parallel factorization control {0: classic algorithm (default), 1:two-level factorization
                            improves when used with many OpenMP threads, 10: improved two-level factorization, caution:
                            this can only be used in conjunction with some certain iparm values for the other iparm
                            parameters
                iparm[24] : Parallel forward/backward solve control
                            {0: use parallel algorithm for solve step (default): use sequential algorithm}
                iparm[25] : Reserved set to zero {0}
                iparm[26] : Matrix checker {0: do not check the sparse matrix representation for errors (default),
                            1: check integer arrays ia and ja. see intel pardiso page}
                iparm[27] : Single/Double precision {0: All double precision (default), 1: all single precision}
                iparm[28] : Reserved, set to zero {0}
                iparm[29] : Output: Number of zero or negative pivots
                iparm[30] : Partial solve: This parameter controls the solve step. It can be used if only few
                            components of solution vector are needed
                            {0: disable (default), 1: Use the perm vector to identify the elements that are zero on RHS,
                            2: Assume only few are nonzero, but make full computation => set rhs to zero values where
                            not needed,
                            3: Just compute selected components defined in the perm vector (interesting option!)
                iparm[31] : Reserved, set to zero {0}
                iparm[32] : Reserved, set to zero {0}
                iparm[33] : Optimal number of OpemMP threads for conditional numerical reproducibility (CNR) mode
                            See https://software.intel.com/en-us/mkl-developer-reference-fortran-pardiso-iparm-parameter
                iparm[34] : {0: One based indexing: columns and rows indexing in arrays ia, ja and perm starts from 1,
                            1: Zero based indexing: columns and rows indexing in arrays ia, ja and perm starts from 0}
                iparm[35] : Schur complement matrix computation control (use perm vector to define that part of matrix
                            that shall be used for Schur complement, {0 (default) compute no Schur complement,
                            1: compute Schur complement and return it in solution vector, 2 see
                            https://software.intel.com/en-us/mkl-developer-reference-fortran-pardiso-iparm-parameter
                iparm[36] : Storage format: {0: Use CSR-Format (default), 1: use BSR format,
                            -1: use variable BSR format}
                iparm[37] : Reserved, set to zero
                iparm[38] : See https://software.intel.com/en-us/mkl-developer-reference-fortran-pardiso-iparm-parameter
                iparm[39] - iparm[54] : Zero
                iparm[55] : Diagonal pivoting control: {0: internal function used to work with pivot and calculation
                            of diagonal arrays turned off (default),
                            1: You can use mkl_pardiso_pivot callback routine to control pivot elements which appear
                            during numerical factorization}
                iparm[56] - iparm[58] : Reserved, set to zero.
                iparm[59] : Intel MKL Pardiso mode: {0: in core mode (default), 1: conditional in core mode (see intel),
                            2: out of core mode (for very large problems where RAM is not sufficient)}
                iparm[60] - iparm[61] : Reserved, set to zero.
                iparm[62] : output: Calculates minimum out of core memory needed after initialization.
                iparm[63] : Reserved, set to zero.

            Returns
            -------
            None

            """

            self.n = A.shape[0]

            self.mtype = None
            self.dtype = np.float64
            self.set_mtype(mtype)

            self.ctypes_dtype = ctypeslib.ndpointer(self.dtype)

            if not isinstance(A, sp.csr_matrix):
                try:
                    A = A.tocsr()
                except Exception:
                    raise ValueError('A must be in csr-format or at least convertible to csr format')

            # If A is symmetric, store only the upper triangular portion
            if self.mtype in [2, -2, 4, -4, 6]:
                A = sp.triu(A, format='csr')

            if not A.has_sorted_indices:
                A.sort_indices()

            self.a = A.data
            self.ia = A.indptr
            self.ja = A.indices

            self._MKL_a = self.a.ctypes.data_as(self.ctypes_dtype)
            self._MKL_ia = self.ia.ctypes.data_as(POINTER(c_int))
            self._MKL_ja = self.ja.ctypes.data_as(POINTER(c_int))

            # Hardcode some parameters for now...
            self.maxfct = 1
            self.mnum = 1
            self.perm = 0

            if verbose:
                self.msglvl = 1
            else:
                self.msglvl = 0

            # Initialize handle to data structure
            self.pt = np.zeros(64, np.int64)
            self._MKL_pt = self.pt.ctypes.data_as(POINTER(c_longlong))

            # Initialize parameters
            self.iparm = np.zeros(64, dtype=np.int32)
            self._MKL_iparm = self.iparm.ctypes.data_as(POINTER(c_int))

            # Initialize pardiso
            pardisoinit(self._MKL_pt, byref(c_int(self.mtype)), self._MKL_iparm)

            # Set some standard iparm values
            self.iparm[1] = 3  # Use parallel nested dissection for reordering
            self.iparm[23] = 1  # Use parallel factorization
            self.iparm[34] = 1  # Zero base indexing

            # Set passed iparms
            if iparm is not None:
                self.set_iparms(iparm)

        def set_mtype(self, mtype):
            """
            Sets the matrix type for pardiso solver.

            Available types are:
                - 1 : real and structurally symmetric (not supported)
                - 2 : real and symmetric positive definite
                - -2 : real and symmetric indefinite
                - 3 : complex and structurally symmetric (not supported)
                - 4 : complex and Hermitian positive definite
                - -4 : complex and Hermitian indefinite
                - 6 : complex and symmetric
                - 11 : real and nonsymmetric (default)
                - 13 : complex and nonsymmetric

            Parameters
            ----------
            mtype: int
                matrix type (integer)

            Returns
            -------

            """
            if mtype in [1, 3]:
                msg = "mtype = 1/3 - structurally symmetric matrices not supported"
                raise NotImplementedError(msg)
            elif mtype in [2, -2, 4, -4, 6, 11, 13]:
                self.mtype = mtype
                if mtype in [4, -4, 6, 13]:
                    # Complex matrix
                    self.dtype = np.complex128
                elif mtype in [2, -2, 11]:
                    # Real matrix
                    self.dtype = np.float64
            else:
                msg = "Invalid mtype: mtype={}".format(mtype)
                raise ValueError(msg)

        def set_iparms(self, iparm):
            """
            Sets parameters for pardiso solver by dict.

            See documentation of constructor for further information.

            Parameters
            ----------
            iparm: dict
                dictionary containing parameters for pardiso solver (c.f. constructor documentation)

            Returns
            -------

            """
            for key in iparm:
                self.iparm[key] = iparm[key]

        def clear(self):
            """
            Clear the memory allocated from the solver.
            """
            self.run_pardiso(phase=-1)

        def factor(self):
            """
            Run the factorization of the solver.

            Returns
            -------

            """
            self.run_pardiso(phase=12)

        def solve(self, rhs):
            """
            Solve method for pardiso solver.
            Factorization and Forward/backward substitution is run in one call.

            Parameters
            ----------
            rhs: numpy.array
                right hand side A x = rhs

            Returns
            -------
            x : numpy.array
                solution to linear equation A x = rhs
            """
            x = self.run_pardiso(phase=33, rhs=rhs)
            return x

        def run_pardiso(self, phase, rhs=None):
            """
            Run specified phase of the Pardiso solver.

            Parameters
            ----------
            phase : int
                Flag setting the analysis type of the solver:

                -  11 : Analysis
                -  12 : Analysis, numerical factorization
                -  13 : Analysis, numerical factorization, solve, iterative refinement
                -  22 : Numerical factorization
                -  23 : Numerical factorization, solve, iterative refinement
                -  33 : Solve, iterative refinement
                - 331 : like phase=33, but only forward substitution
                - 332 : like phase=33, but only diagonal substitution (if available)
                - 333 : like phase=33, but only backward substitution
                -   0 : Release internal memory for L and U matrix number mnum
                -  -1 : Release all internal memory for all matrices
            rhs : ndarray, optional
                Right hand side of the equation `A x = rhs`. Can either be a vector
                (array of dimension 1) or a matrix (array of dimension 2). Default
                is None.

            Returns
            -------
            x : ndarray
                Solution of the system `A x = rhs`, if `rhs` is provided. Is either
                a vector or a column matrix.

            """

            if rhs is None:
                nrhs = 0
                rhs = np.zeros(1)
            else:
                if rhs.ndim == 1:
                    nrhs = 1
                elif rhs.ndim == 2:
                    nrhs = rhs.shape[1]
                else:
                    msg = "Right hand side must either be a 1 or 2 dimensional " \
                          "array. Higher order right hand sides are not supported."
                    raise NotImplementedError(msg)

            shape_out = rhs.shape
            rhs = rhs.astype(self.dtype).flatten(order='f')
            x = np.zeros(nrhs*self.n, dtype=self.dtype)

            MKL_rhs = rhs.ctypes.data_as(self.ctypes_dtype)
            MKL_x = x.ctypes.data_as(self.ctypes_dtype)
            ERR = 0

            pardiso(self._MKL_pt,               # pt
                    byref(c_int(self.maxfct)),  # maxfct
                    byref(c_int(self.mnum)),    # mnum
                    byref(c_int(self.mtype)),   # mtype
                    byref(c_int(phase)),        # phase
                    byref(c_int(self.n)),       # n
                    self._MKL_a,                # a
                    self._MKL_ia,               # ia
                    self._MKL_ja,               # ja
                    byref(c_int(self.perm)),    # perm
                    byref(c_int(nrhs)),         # nrhs
                    self._MKL_iparm,            # iparm
                    byref(c_int(self.msglvl)),  # msglvl
                    MKL_rhs,                    # b
                    MKL_x,                      # x
                    byref(c_int(ERR)))          # error

            if len(shape_out) == 2:
                x = x.reshape((self.n, nrhs), order='f')
            return x

except Exception as e:
    raise e
