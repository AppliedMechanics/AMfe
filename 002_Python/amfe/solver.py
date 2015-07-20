# -*- coding: utf-8 -*-

'''
Module for solving static and dynamic problems.

'''

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

def norm_of_vector(array):
    return np.sqrt(array.T.dot(array))

class NewmarkIntegrator():
    '''
    Newmark-integration scheme using generalized alpha routine for nonlinear second order systems

    Parameters
    -----------
    alpha : float, optional
        damping factor of the generalized alpha routine for numerical damping
    vebose : bool, optional
        flag for making the integration process verbose, i.e. printing the residual for every correction in the newton iteration
    n_iter_max : int, optional
        number of maximum iteration in the newton correction process

    Notes
    ------
    This integration scheme is an unconditionally stable implicit nonlinear integration scheme. The unconditional stabiltiy refers to the stability of the solution itself, not on the solutin procedure. With a too tight tolerance (eps > 1E8) the solution might not converge, as the solution procedure can not lower the residual below a threshold that's higher than the eps. In general the solution should converge in less than ten iteration steps.

    Examples
    --------
    TODO

    References:
    -----------
    M. Géradin and D. J. Rixen. Mechanical vibrations: theory and application
    to structural dynamics. John Wiley & Sons, 2014. pp. 564.

    '''

    def __init__(self, alpha=0, verbose=False, n_iter_max=40):
        self.beta = 1/4*(1 + alpha)**2
        self.gamma = 1/2 + alpha
        self.delta_t = 1E-3
        self.eps = 1E-8
        self.newton_damping = 0.8
        self.residual_threshold = 1E6
        self.mechanical_system = None
        self.verbose = verbose
        self.n_iter_max = n_iter_max
        pass

    def set_nonlinear_model(self, f_non, K, M, f_ext=None):
        '''
        Sets the nonlinear model with explicit function metioning.

        Parameters
        -----------
        f_non : function
            function of nonlinear force being called with f_non(q) and returning the nonlinear force as ndarray
        K : function
            function of tangential stiffness matrix being called with K(q) and returning the nonlinear stiffness matrix as ndarray
        M : ndarray
            mass matrix
        f_ext : function, optional
            function of external force being called with f_ext(q, dq, t) and returning the external force as ndarray

        Examples
        ---------

        Notes
        ------
        This function serves basically as a test function. For elaborate finite
        element work the set_mechanical_system interface is more convenient and
        does everything automatically including the recording of displacements etc.

        See Also:
        ---------
        set_mechanical_system

        '''
        # decorator for the efficient computation of the tangential stiffness
        # matrix and force in one step; Basic intention is to make assembly process only once.
        def K_and_f_non(q):
            return K(q), f_non(q)
        self.K_and_f_non = K_and_f_non
        self.f_non = f_non
        self.M = M
        self.f_ext = f_ext

    def set_mechanical_system(self, mechanical_system):
        '''
        hands over the mechanical system as a whole to the integrator.
        The matrices for the integration routine are then taken right from the mechanical system

        Parameters
        -----------
        mechanical_system : MechanicalSystem
            instance of MechanicalSystem which should be integrated

        Returns
        --------
        None

        Examples
        ---------
        TODO

        '''
        self.mechanical_system = mechanical_system
        self.M = mechanical_system.M_global()
#        self.K = mechanical_system.K_global
        self.f_non = mechanical_system.f_int_global
        self.f_ext = mechanical_system.f_ext_global
        self.K_and_f_non = mechanical_system.K_and_f_global


    def _residual(self, f_non, q, dq, ddq, t):
        '''computes the residual of the system with the given variables'''
        if self.f_ext is not None:
            res = self.M.dot(ddq) + f_non - self.f_ext(q, dq, t)
        else:
            res = self.M.dot(ddq) + f_non
        return res


    def integrate_nonlinear_system(self, q_start, dq_start, time_range):
        '''
        Integrates the system using generalized alpha method.

        Parameters
        -----------
        q_start : ndarray
            initial displacement of the constrained system in voigt notation
        dq_start : ndarray
            initial velocity of the constrained system in voigt notation
        time_range : ndarray
            vector containing the time points at which the state is written to the output

        Returns
        --------
        q : ndarray
            displacements of the system with q[:, i] being the
            displacement of the i-th timestep in voigt notation
        dq : ndarray
            velocities of the system with dq[:, i] being the velocity of the
            i-th timestep in voigt notation

        Examples
        ---------
        TODO

        '''
        # initialize starting variables
        q = q_start.copy()
        dq = dq_start.copy()
        ddq = np.zeros(len(q_start))

        q_global = []
        dq_global = []
        t = 0
        time_index = 0 # index of the timestep in the time_range array
        write_flag = False
        # catch start value 0:
        if time_range[0] < 1E-12:
            q_global.append(q)
            dq_global.append(dq)
            time_index = 1
        else:
            time_index = 0

        # predict start values for ddq:
        ddq = linalg.spsolve(self.M, self.f_non(q))
        no_newton_convergence_flag = False
        while time_index < len(time_range):
            # time tolerance fitting...
            if t + self.delta_t + 1E-8 >= time_range[time_index]:
                dt = time_range[time_index] - t
                if dt < 1E-8:
                    dt = 1E-7
                write_flag = True
                time_index += 1
            else:
                dt = self.delta_t
                if no_newton_convergence_flag:
                    dt /= 2
                    no_newton_convergence_flag = False
            # Handling if no convergence is gained:
            t_old = t
            q_old = q.copy()
            dq_old = dq.copy()

            t += dt
            # Prediction
            q += dt*dq + (1/2-self.beta)*dt**2*ddq
            dq += (1-self.gamma)*dt*ddq
            ddq *= 0

            # checking residual and convergence
            K, f_non = self.K_and_f_non(q)
            res = self._residual(f_non, q, dq, ddq, t)
            res_abs = norm_of_vector(res)

            # Newcton-Correction-loop
            n_iter = 0
            while res_abs > self.eps*norm_of_vector(f_non):
                S = K + 1/(self.beta*dt**2)*self.M
                delta_q = - linalg.spsolve(S, res)
                if res_abs > self.residual_threshold:
                    delta_q *= self.newton_damping
                q   += delta_q
                dq  += self.gamma/(self.beta*dt)*delta_q
                ddq += 1/(self.beta*dt**2)*delta_q
                K, f_non = self.K_and_f_non(q)
                res = self._residual(f_non, q, dq, ddq, t)
                res_abs = norm_of_vector(res)
                n_iter += 1
                if self.verbose:
                    print('Iteration', n_iter, 'Residuum:', res_abs)
                # catch when the newton loop doesn't converge
                if n_iter > self.n_iter_max:
                    t = t_old
                    q = q_old.copy()
                    dq = dq_old.copy()
                    no_newton_convergence_flag = True
                    break
                pass

            print('Zeit:', t, 'Anzahl an Iterationen:', n_iter, 'Residuum:', res_abs)
            # Writing if necessary:
            if write_flag:
                # writing to the mechanical system, if possible
                if self.mechanical_system:
                    self.mechanical_system.write_timestep(t, q)
                else:
                    q_global.append(q.copy())
                    dq_global.append(dq.copy())
                write_flag = False
            pass # end of time loop

        return np.array(q_global), np.array(dq_global)



def solve_linear_displacement(mechanical_system, t=0, verbose=True):
    '''
    Solve the linear static problem of the mechanical system and print
    the results directly to the mechanical system

    Parameters
    ----------
    mechanical_system :   Instance of the class MechanicalSystem

    t : float
        time for the external force call in MechanicalSystem

    Returns
    -------
    None

    '''
    f_ext = mechanical_system.f_ext_global(None, None, t)
    mechanical_system.write_timestep(0, f_ext*0) # write zeros

    if verbose: print('Start solving linear static problem')

    u = linalg.spsolve(mechanical_system.K_global(), f_ext)
    mechanical_system.write_timestep(1, u)

    if verbose: print('Static problem solved')

    pass

def solve_nonlinear_displacement(mechanical_system, no_of_load_steps=10,
                                 t=0, eps=1E-12, newton_damping=1,
                                 n_max_iter=1000, smplfd_nwtn_itr=1, verbose=True):
    '''
    Solver for the nonlinear system applied directly on the mechanical system.

    Prints the results directly to the mechanical system

    Parameters
    ----------
    mechanical_system : MechanicalSystem
        Instance of the class MechanicalSystem
    no_of_load_steps : int
        Number of equally spaced load steps which are applied in order to receive the solution
    t : float, optional
        time for the external force call in mechanical_system
    eps : float, optional
        Epsilon for assessment, when a loadstep has converged
    newton_damping : float, optional
        Newton-Damping factor applied in the solution routine; 1 means no damping,
        0 < newton_damping < 1 means damping
    n_max_iter : int, optional
        Maximum number of interations in the Newton-Loop
    smplfd_nwtn_itr : int, optional
          Number at which the jacobian is updated; if 1, then a full newton scheme is applied;
          if very large, it's a fixpoint iteration with constant jacobian
    verbose : bool, optional
        print messages if necessary

    Returns
    -------
    None

    Examples
    ---------
    TODO

    '''
    stepwidth = 1/no_of_load_steps
    f_ext = mechanical_system.f_ext_global(None, None, t)
    ndof = f_ext.shape[0]
    u = np.zeros(ndof)
    mechanical_system.write_timestep(0, u) # initial write

    abs_f_ext = np.sqrt(f_ext.dot(f_ext))
    for force_factor in np.arange(stepwidth, 1+stepwidth, stepwidth):
        # prediction
        K, f_int= mechanical_system.K_and_f_global(u)
        res = f_int - f_ext*force_factor
        abs_res = norm_of_vector(res)

        # Newton-Loop
        n_iter = 0
        while (abs_res > eps*abs_f_ext) and (n_max_iter > n_iter):
            corr = linalg.spsolve(K, res)
            u -= corr*newton_damping
            K, f_int= mechanical_system.K_and_f_global(u)
            res = f_int - f_ext * force_factor
            abs_res = norm_of_vector(res)
            n_iter += 1
            if verbose: print('Stufe', force_factor, 'Iteration Nr.', n_iter, \
                                'Residuum:', abs_res)

        mechanical_system.write_timestep(force_factor, u)     


def give_mass_and_stiffness(mechanical_system):
    '''
    Determine mass and stiffness matrix of a mechanical system

    Parameters
    ----------
    mechanical_system : MechanicalSystem
        Instance of the class MechanicalSystem

    Returns
    -------
    M : ndarray
        Mass matrix of the mechanical system
    K : ndarray
        Stiffness matrix of the mechanical system
    '''

    K = mechanical_system.K_global()
    M = mechanical_system.M_global()
    return M, K
