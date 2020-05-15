# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Module contains linear equation solvers
"""

from scipy.linalg import solve as scipysolve
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import spsolve, cg
from copy import deepcopy
import numpy as np
import logging


__all__ = [
    'ScipySparseLinearSolver',
    'ScipyConjugateGradientLinearSolver',
    'PardisoLinearSolver',
    'ResidualbasedConjugateGradient',
    'solve_sparse'
]


class LinearSolverBase:
    """
    Base Class for all linear solvers
    """
    def __init__(self):
        pass

    def solve(self, A, b):
        """
        Solve a linear system A x = b
        Parameters
        ----------
        A : ndarray
            ndarray desribing the matrix A
        b : ndarray
            right hand side

        Returns
        -------
        x : ndarray
            Solution vector
        """
        pass


# Wrappers for third-party linear solvers
class ScipySparseLinearSolver(LinearSolverBase):
    """
    Scipy Sparse Solver

    Notes
    -----
    This tool uses the Intel MKL library provided by Anaconda. If the Intel MKL
    is not installed, especially for large systems the computation time can go
    crazy. To adjust the number of threads used for the computation, it is
    recommended to use the mkl-service module provided by Anaconda:

    >>> import mkl
    >>> mkl.get_max_threads()
    2
    >>> mkl.set_num_threads(1)
    >>> mkl.get_max_threads()
    1

    """
    AVAILABLE_OPTIONS = {'permc_spec': 'How to permute the columns of the matrix for sparsity preservation'
                                       'Allowed Values: NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD',
                         'use_umfpack': 'True or False for using umfpack. This can only be done if scikit-umfpack'
                                        'is installed'
                         }

    def __init__(self):
        super().__init__()

    def solve(self, A, b, **kwargs):
        """
        Solve a linear system A x = b
        Parameters
        ----------
        A : {ndarray, matrix, LinearOperator}
            ndarray desribing the matrix A
        b : ndarray
            right hand side

        Returns
        -------
        x : ndarray
            Solution vector
        """
        x = spsolve(A, b, **kwargs)
        return x


class ScipyConjugateGradientLinearSolver(LinearSolverBase):
    def __init__(self):
        super().__init__()

    def solve(self, A, b, x0=None, tol=1e-05, maxiter=None, P=None, callback=None, atol=None):
        """

        Parameters
        ----------
        A : {sparse_matrix, dense_matrix, LinearOperator}
            Matrix A
        b : {array, matrix}
            Right hand side
        x0 : {array, matrix}
            starting guess for the solution
        tol, atol : float, optional
            Tolerances for convergence norm(residual) <= max(tol*norm(b), atol)
        maxiter : int
            maximum number of iterations
        P : {sparse_matrix, dense_matrix, LinearOperator}
            Preconditioner for A. The preconditioner should approximate the inverse of A
        callback : function
            User-supplied function to call after each iteration.
            Signature callback(xk), where xk is the current solution vector

        Returns
        -------
        x : {array, matrix}
            solution vector
        """
        x, _ = cg(A, b, x0, tol, maxiter, P, callback, atol)
        return x


try:
    from .lib import PardisoWrapper

    class PardisoLinearSolver(LinearSolverBase):

        MTYPES = {'sym': 1,
                  'spd': 2,
                  'sid': -2,
                  'nonsym': 11,
                  }

        # info:
        # For changing iparms that are not listed here, just add a name for the iparm parameter
        # Then you can pass an options dictionary to change the iparms
        IPARM_DICT = {'refinement_steps': 7,
                      'pivoting_perturbation': 9,
                      'scaling': 10,
                      'transposed': 11,
                      'maximum_weighted_matching': 12,
                      'indefinite_pivoting': 20,
                      'partial_solve': 30,
                      'storage_mode': 59,
                      }

        def __init__(self):
            super().__init__()
            self.wrapper_class = None

        def solve(self, A, b, mtype='nonsym', **iparms):
            """

            Parameters
            ----------
            A : csr_matrix or ndarray
                Matrix A
            b : ndarray
                Right hand side
            mtype : {'sid', 'sym', 'spd', 'nonsym'}
                Matrix type (symmetric indefinite, symmetric, symmetric positive definite, nonsymmetric)
            iparms : dict
                e.g. {'transposed': 1, 'scaling': 1}

            Returns
            -------
            x : ndarray
                solution
            """
            A = csr_matrix(A)
            # Notes:
            # saddle point problem: use iparms: scaling and maximum_weighted_matching
            self.wrapper_class = PardisoWrapper(A, mtype=self.MTYPES[mtype], iparm=self._parse_iparms(iparms))

            # Notes:
            # Check if wrapper_class object is already factorized
            # return self.wrapper_class.solve(b)
            # Solve in one step
            x = self.wrapper_class.run_pardiso(13, b)
            self.wrapper_class.clear()
            return x

        def _parse_iparms(self, iparms):
            return dict([(self.IPARM_DICT[key], iparms[key]) for key in iparms])

except Exception as e:

    logger = logging.getLogger(__name__)
    logger.warning('PardisoLinearSolver could not be loaded. Possibly mkllib is not installed properly')

    class PardisoLinearSolver:
        def __init__(self, *args, **kwargs):
            raise ImportError('PardisoLinearSolver is not available on your system,'
                              'probably because mkllib could not be found.'
                              'If you use anaconda distribution, try: conda install mkl')


# Own solvers
class ResidualbasedConjugateGradient:
    """
    Conjugate Gradient-solver, which solves a linear system of equations
    
    A*x = b
    
    for x. Instead of handing over A and b, a callback of the linear problem's current residual is required,
    i.e., res = A*x-b.
    This might be more convenient in certain cases, where it's more implementation-friendly to evaluate the residual
    instead of building A and b explicitly.
    """
    
    def __init__(self):
        super().__init__()
        
    def solve(self, residual_callback, x0, tol=1e-05, maxiter=None):
        """
        Solver-method of the residual-based Conjugate-Gradient iterative solver. 
        
        Parameters
        ----------
        residual_callback : method
            callback-method of the linear system's residual, which is updated by the solution 'x'.
            Hence the method has to be of the form 'residual(x)'.
        x0 : ndarray
            starting guess for the solution
        tol: float, optional
            Tolerance for convergence norm(residual) <= max(tol*norm(b), atol)
        maxiter : int
            maximum number of iterations

        Returns
        -------
        x : {array, matrix}
            solution vector
        """

        logger = logging.getLogger(__name__)
        sol = x0
        d = residual_callback(np.zeros(sol.shape))
        res = residual_callback(sol)
        
        w = deepcopy(res)

        # Standard Conjugate Gradient
        conv_crit = np.linalg.norm(res)

        cg_iter = 1
        converged = False
        if conv_crit <= tol:
            converged = True
            logger.debug("CG converged due to initial residual=0")
        while not converged:
            q = -residual_callback(w) + d
                            
            alpha = np.dot(res.T, res) / np.dot(w.T, q)
            
            sol += alpha * w
            
            res_old = deepcopy(res)
            res -= alpha * q
            
            conv_crit = np.linalg.norm(res)
            if conv_crit <= tol:
                converged = True
                logger.debug("CG converged at iteration {};  Residual: {}".format(cg_iter, conv_crit))
                break
            
            if cg_iter >= maxiter or conv_crit > 1e6:
                logger.debug("WARNING: CG not converged at iteration {};  Residual: {}".format(cg_iter, conv_crit))
                break
            
            beta = np.dot(res.T, res) / np.dot(res_old.T, res_old)
            
            w = res + beta * w

            cg_iter += 1
            logger.debug("CG iteration {};  Residual: {}".format(cg_iter, conv_crit))
            
        residual_callback(sol)
            
        return sol, converged, cg_iter


def solve_sparse(A, b, matrix_type='sid'):
    r"""
    Abstraction of the solution of the sparse system Ax=b using the fastest
    solver available for sparse and non-sparse matrices.

    Parameters
    ----------
    A : sp.sparse.CSR
        sparse matrix in CSR-format
    b : ndarray
        right hand side of equation
    matrix_type : {'spd', 'sym', 'nonsym', 'sid'}, optional
        Specifier for the matrix type:

        - 'spd' : symmetric positive definite
        - 'sid' : symmetric indefinite, default.
        - 'nonsym' : generally unsymmetric

    Returns
    -------
    x : ndarray
        solution of system Ax=b

    Notes
    -----
    This tool uses the Intel MKL library provided by Anaconda. If the Intel MKL
    is not installed, especially for large systems the computation time can go
    crazy. To adjust the number of threads used for the computation, it is
    recommended to use the mkl-service module provided by Anaconda:

    >>> import mkl
    >>> mkl.get_max_threads()
    2
    >>> mkl.set_num_threads(1)
    >>> mkl.get_max_threads()
    1

    """
    if issparse(A):
        # if use_pardiso:
        try:
            solver = PardisoLinearSolver()
            x = solver.solve(A, b, matrix_type)
        except Exception:
            x = scipysolve(A, b)
        # else:
        # use scipy solver instead
        # x = spsolve(A, b)
    else:
        x = scipysolve(A, b)
    return x
