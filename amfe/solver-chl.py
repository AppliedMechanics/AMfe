def solve_nonlinear_dynamics_state_space(
        mechanical_system, x0, time_range, dt,
        scheme = 'jwh_alpha',
        rho_inf=0.9,
        rtol=1.0E-9,
        atol=1.0E-6,
        verbose=False,
        n_iter_max=30,
        conv_abort=True,
        write_iter=False,
        track_niter=True,
        matrix_type='unsymm'):
    '''
    Time integration of the non-linear state-space system.

    Parameters
    ----------
    mechanical_system : instance of MechanicalSystemStateSpace
        State-space ystem to be integrated.
    x0 : ndarray
        Initial state = start displacement (q0) and start velocity (dq0).
    time_range : ndarray
        Array of discrete timesteps, at which the solution is saved.
    dt : float
        Time step size of the integrator.
    scheme : string, optional
        Time integration scheme. Default scheme: jwh_alpha.
    rho_inf : float, optional
        high-frequency spectral radius, has to be 0 <= rho_inf <= 1. For 1 no
        damping is apparent, for 0 maximum damping is there. Default value: 0.9
    rtol : float, optional
        Relative tolerance with respect to the maximum external force for the
        Newton-Raphson iteration. Default value: 1E-8.
    atol : float, optional
        Absolute tolerance for the Newton_Raphson iteration.
        Default value: 1E-6.
    verbose : bool, optional
        Flag setting verbose output. Default: False.
    n_iter_max : int, optional
        Number of maximum iterations per Newton-Raphson-procedure. Default
        value is 30.
    conv_abort : bool, optional
        Flag setting, if time integration is aborted in the case when no
        convergence is gained in the Newton-Raphson-Loop. Default value is
        True.
    write_iter : bool, optional
        Flag setting, if every step of the Newton-Raphson iteration is written
        to the MechanicalSystem object. Useful only for debugging, when no
        convergence is gained. Default value: False.
    track_niter : bool, optional
        Flag for the iteration-count. If True, the number of iterations in the
        Newton-Raphson-Loop is counted and saved to iteration_info in the
        mechanical system.

    References
    ----------
       [1]  J. Chung and G. Hulbert (1993): A time integration algorithm for structural
            dynamics with improved numerical dissipation: the generalized-alpha method.
            Journal of Applied Mechanics 60(2) 371--375.
       [2]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha
            method for integrating the filtered Navier-Stokes equations with a
            stabilized finite element method. Computer Methods in Applied Mechanics and
            Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
       [3]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the
            first-order generalised-alpha scheme for structural dynamic problems.
            Computers and Structures 193 226--238. DOI 10.1016/j.compstruc.2017.08.013.
       [4]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and
            application to structural dynamics. ISBN 978-1-118-90020-8.

    '''
    t_clock_1 = time.time()

    # prepare mechanical_system
    mechanical_system.clear_timesteps()
    mechanical_system.iteration_info = []

    # initialize variables
    t = time_range[0]
    x = x0.copy()
    dx = np.zeros_like(x0)
    F_ext = np.zeros_like(x0)
    abs_F_ext = atol
    time_index = 0

    # set parameters
    eps = 1E-13
    eval('mechanical_system.set_parameters_' + scheme)(dt, rho_inf)
    predict = eval('mechanical_system.predict_' + scheme)
    newton_raphson = eval('mechanical_system.newton_raphson_' + scheme)
    update = eval('mechanical_system.update_' + scheme)

    # time step loop
    while time_index < len(time_range):

        # write output
        if t + eps >= time_range[time_index]:
            mechanical_system.write_timestep(t, x.copy())
            time_index += 1
            if time_index == len(time_range):
                break

        # save old variables
        x_old = x.copy()
        dx_old = dx.copy()
        F_ext_old = F_ext.copy()
        t_old = t

        # predict new variables using old ones
        t += dt
        predict(x, dx)

        Jac, res, F_ext = newton_raphson(x, dx, x_old, dx_old, t, t_old)

        abs_F_ext = max(abs_F_ext, norm_of_vector(F_ext))
        res_abs = norm_of_vector(res)

        # Newton-Raphson iteration loop
        n_iter = 0
        while res_abs > rtol*abs_F_ext + atol:

            if sp.sparse.issparse(Jac):
                delta_x = - solve_sparse(Jac, res, matrix_type=matrix_type)
            else:
                delta_x = - sp.linalg.solve(Jac, res)

            # update variables
            update(x, dx, delta_x)

            # update system matrices and vectors
            Jac, res, F_ext = newton_raphson(x, dx, x_old, dx_old, t, t_old)

            res_abs = norm_of_vector(res)
            n_iter += 1

            if verbose:
                if sp.sparse.issparse(Jac):
                    cond_nr = 0.0
                else:
                    cond_nr = np.linalg.cond(Jac)
                print(('Iteration: {0:3d}, residual: {1:6.3E}, condition# of Jacobian: '
                       + '{2:6.3E}').format(n_iter, res_abs, cond_nr))

            # write state
            if write_iter:
                t_write = t + dt/1000*n_iter
                mechanical_system.write_timestep(t_write, x.copy())

            # catch failing Newton-Raphson iteration
            if n_iter > n_iter_max:
                if conv_abort:
                    print(abort_statement)
                    mechanical_system.iteration_info = np.array(
                            mechanical_system.iteration_info)
                    t_clock_2 = time.time()
                    print('Time for time marching integration: '
                          + '{0:6.3f}s.'.format(t_clock_2 - t_clock_1))
                    return

                t = t_old
                x = x_old.copy()
                F_ext = F_ext_old.copy()
                break

            # end of Newton-Raphson iteration loop

        print(('Time: {0:3.6f}, #iterations: {1:3d}, '
               + 'residual: {2:6.3E}').format(t, n_iter, res_abs))
        if track_niter:
            mechanical_system.iteration_info.append((t, n_iter, res_abs))

        # end of time step loop

    # write iteration info to mechanical system
    mechanical_system.iteration_info = np.array(mechanical_system.iteration_info)

    # measure integration end time
    t_clock_2 = time.time()
    print('Time for time marching integration: {0:6.3f} seconds'.format(
          t_clock_2 - t_clock_1))
    return


def solve_linear_dynamics(
        mechanical_system, q0, dq0, time_range, dt,
        scheme='generalized_alpha',
        rho_inf=0.9):
    '''
    Time integration of the linearized second-order system.

    Parameters
    ----------
    mechanical_system : instance of MechanicalSystem
        Mechanical system to be linearized at zero displacement and inetgrated.
    q0 : ndarray
        Initial displacement.
    dq0 : ndarray
        Initial velocity.
    time_range : ndarray
        Array containing the time steps to be exported.
    dt : float
        Time step size.
    scheme : string, optional
        Time integration scheme. Default scheme: generalized_alpha.
    rho_inf : float, optional
        High-frequency spectral radius, has to be 0 <= rho_inf <= 1. For 1 no
        damping is apparent, for 0 maximum damping is there. Default value: 0.9.

    References
    ----------
       [1]  J. Chung and G. Hulbert (1993): A time integration algorithm for structural
            dynamics with improved numerical dissipation: the generalized-alpha method.
            Journal of Applied Mechanics 60(2) 371--375.
       [2]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha
            method for integrating the filtered Navier-Stokes equations with a
            stabilized finite element method. Computer Methods in Applied Mechanics and
            Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
       [3]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the
            first-order generalised-alpha scheme for structural dynamic problems.
            Computers and Structures 193 226--238. DOI 10.1016/j.compstruc.2017.08.013.
       [4]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and
            application to structural dynamics. ISBN 978-1-118-90020-8.

    '''
    t_clock_1 = time.time()

    # prepare mechanical_system
    mechanical_system.clear_timesteps()

    # initialize variables
    t = time_range[0]
    q = q0.copy()
    dq = dq0.copy()
    if scheme is 'jwh_alpha':
        v = dq0.copy()
    else:
        v = np.empty((0,0))
    ddq = np.zeros_like(q0)
    time_index = 0

    # set functions and parameters
    eps = 1E-13
    eval('mechanical_system.set_parameters_' + scheme)(dt, rho_inf)
    effective_force = eval('mechanical_system.effective_force_' + scheme)
    post_process = eval('mechanical_system.post_process_' + scheme)

    # LR-decompose effective stiffness
    K_eff = eval('mechanical_system.effective_stiffness_' + scheme)()
    if sp.sparse.issparse(K_eff):
        K_eff_inv = SpSolve(K_eff, matrix_type='symm')
    else:
        K_eff_inv = sp.linalg.lu_factor(a=K_eff, overwrite_a=True, check_finite=False)

    # evaluate initial acceleration
    ddq = solve_sparse(mechanical_system.M_constr,
                       mechanical_system.f_ext(q, dq, t) \
                       - mechanical_system.D_constr@dq \
                       - mechanical_system.K_constr@q)

    # time step loop
    while time_index < len(time_range):

        # write output
        if t + eps >= time_range[time_index]:
            mechanical_system.write_timestep(t, q.copy())
            time_index += 1
            if time_index == len(time_range):
                break

        # save old variables
        q_old = q.copy()
        dq_old = dq.copy()
        v_old = v.copy()
        ddq_old = ddq.copy()
        t_old = t

        # solve system
        t += dt
        f_eff = effective_force(q_old, dq_old, v_old, ddq_old, t, t_old)
        if sp.sparse.issparse(K_eff):
            q = K_eff_inv.solve(f_eff)
        else:
            q = sp.linalg.lu_solve(lu_and_piv=K_eff_inv, b=f_eff, trans=0, \
                                   overwrite_b=True, check_finite=False)

        # update variables
        dq, v, ddq = post_process(q, q_old, dq_old, v_old, ddq_old)
        print('Time: {0:3.6f}'.format(t))

        # end of time step loop

    if sp.sparse.issparse(K_eff):
        K_eff_inv.clear()

    # measure integration end time
    t_clock_2 = time.time()
    print('Time for linear time marching integration: {0:6.3f} seconds'.format(
       t_clock_2 - t_clock_1))
    return


def solve_linear_dynamics_state_space(
        mechanical_system, x0, time_range, dt,
        scheme='jwh_alpha',
        rho_inf=0.9):
    '''
    Time integration of the linearized state-space system.

    Parameters
    ----------
    mechanical_system : instance of MechanicalSystemStateSpace
        State-space system to be linearized at the zero displacement and integrated.
    x0 : ndarray
        Initial state = start displacement (q0) and start velocity (dq0).
    time_range : ndarray
        Array containing the time steps to be exported.
    dt : float
        Time step size.
    scheme : string, optional
        Time integration scheme. Default scheme: jwh_alpha.
    rho_inf : float, optional
        High-frequency spectral radius, has to be 0 <= rho_inf <= 1. For 1 no
        damping is apparent, for 0 maximum damping is there. Default value: 0.9

    References
    ----------
       [1]  J. Chung and G. Hulbert (1993): A time integration algorithm for structural
            dynamics with improved numerical dissipation: the generalized-alpha method.
            Journal of Applied Mechanics 60(2) 371--375.
       [2]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha
            method for integrating the filtered Navier-Stokes equations with a
            stabilized finite element method. Computer Methods in Applied Mechanics and
            Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
       [3]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the
            first-order generalised-alpha scheme for structural dynamic problems.
            Computers and Structures 193 226--238. DOI 10.1016/j.compstruc.2017.08.013.
       [4]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and
            application to structural dynamics. ISBN 978-1-118-90020-8.

    '''
    t_clock_1 = time.time()

    # prepare mechanical_system
    mechanical_system.clear_timesteps()

    # initialize variables
    t = time_range[0]
    x = x0.copy()
    dx = np.zeros_like(x0)
    time_index = 0

    # set functions and parameters
    eps = 1E-13
    eval('mechanical_system.set_parameters_' + scheme)(dt, rho_inf)
    effective_force = eval('mechanical_system.effective_force_' + scheme)
    post_process = eval('mechanical_system.post_process_' + scheme)

    # LR-decompose effective stiffness
    K_eff = eval('mechanical_system.effective_stiffness_' + scheme)()
    if sp.sparse.issparse(K_eff):
        K_eff_inv = SpSolve(K_eff, matrix_type='unsymm')
    else:
        K_eff_inv = sp.linalg.lu_factor(a=K_eff, overwrite_a=True, check_finite=False)

    # evaluate initial derivative
    dx = solve_sparse(mechanical_system.E_constr, \
                      mechanical_system.A_constr@x + mechanical_system.F_ext(x, t))

    # time step loop
    while time_index < len(time_range):

        # write output
        if t + eps >= time_range[time_index]:
            mechanical_system.write_timestep(t, x.copy())
            time_index += 1
            if time_index == len(time_range):
                break

        # save old variables
        x_old = x.copy()
        dx_old = dx.copy()
        t_old = t

        # solve system
        t += dt
        F_eff = effective_force(x_old, dx_old, t, t_old)
        if sp.sparse.issparse(K_eff):
            x = K_eff_inv.solve(F_eff)
        else:
            x = sp.linalg.lu_solve(lu_and_piv=K_eff_inv, b=F_eff, trans=0, \
                                   overwrite_b=True, check_finite=False)

        # update variables
        dx = post_process(x, x_old, dx_old)
        print('Time: {0:3.6f}'.format(t))

        # end of time step loop

    if sp.sparse.issparse(K_eff):
        K_eff_inv.clear()

    # measure integration end time
    t_clock_2 = time.time()
    print('Time for linear time marching integration: {0:6.3f} seconds'.format(
       t_clock_2 - t_clock_1))
    return


def solve_nonlinear_statics(
        mechanical_system, no_of_load_steps=10,
        t=0, rtol=1E-8, atol=1E-14, newton_damping=1,
        n_max_iter=1000,
        smplfd_nwtn_itr=1,
        verbose=True,
        track_niter=False,
        write_iter=False,
        conv_abort=True,
        save=True):
    '''
    Solves the non-linear static problem of the mechanical system.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be solved.
    no_of_load_steps : int
        Number of equally spaced load steps which are applied in order to
        receive the solution
    t : float, optional
        time for the external force call in mechanical_system
    rtol : float, optional
        Relative tolerance to external force for estimation, when a loadstep
        has converged
    atol : float, optional
        Absolute tolerance for estimation, of loadstep has converged
    newton_damping : float, optional
        Newton-Damping factor applied in the solution routine; 1 means no damping,
        0 < newton_damping < 1 means damping
    n_max_iter : int, optional
        Maximum number of interations in the Newton-Loop
    smplfd_nwtn_itr : int, optional
          Number at which the jacobian is updated; if 1, then a full newton
          scheme is applied; if very large, it's a fixpoint iteration with
          constant jacobian
    verbose : bool, optional
        print messages if necessary
    track_niter : bool, optional
        Flag for the iteration-count. If True, the number of iterations in the
        Newton-Raphson-Loop is counted and saved to iteration_info in the
        mechanical system.
    write_iter : bool, optional
        Flag setting, if every step of the Newton-Raphson iteration is written
        to the MechanicalSystem object. Useful only for debugging, when no
        convergence is gained. Default value: False.
    conv_abort : bool, optional
        Flag setting, if time integration is aborted in the case when no
        convergence is gained in the Newton-Raphson-Loop. Default value is
        True.
    save : bool, optional
        Write the resulting load steps to the MechanicalSystem to export
        it afterwards. Default value: True


    Returns
    -------
    u : ndarray, shape(ndim, no_of_load_steps)
        Solution displacements; u[:,-1] is the last displacement

    Examples
    ---------
    TODO

    '''
    t_clock_1 = time.time()
    iteration_info = [] # List tracking the number of iterations
    mechanical_system.clear_timesteps()

    u_output = []
    stepwidth = 1/no_of_load_steps
    K, f_int= mechanical_system.K_and_f()
    ndof = K.shape[0]
#   Does not work for reduced systems
#     ndof = mechanical_system.dirichlet_class.no_of_constrained_dofs
    u = np.zeros(ndof)
    du = np.zeros(ndof)
    mechanical_system.write_timestep(0, u) # initial write

    for t in np.arange(stepwidth, 1+stepwidth, stepwidth):

        # prediction
        K, f_int= mechanical_system.K_and_f(u, t)
        f_ext = mechanical_system.f_ext(u, du, t)
        res = - f_int + f_ext
        abs_res = norm_of_vector(res)
        abs_f_ext = np.sqrt(f_ext @ f_ext)

        # Newton-Loop
        n_iter = 0
        while (abs_res > rtol*abs_f_ext + atol) and (n_max_iter > n_iter):
            corr = solve_sparse(K, res)
            u += corr*newton_damping
            if (n_iter % smplfd_nwtn_itr) is 0:
                K, f_int = mechanical_system.K_and_f(u, t)
                f_ext = mechanical_system.f_ext(u, du, t)
            res = - f_int + f_ext
            abs_f_ext = np.sqrt(f_ext @ f_ext)
            abs_res = norm_of_vector(res)
            n_iter += 1

            if verbose:
                print(('Step: {0:3d}, iteration#: {1:3d}'
                      + ', residual: {2:6.3E}').format(int(t), n_iter, abs_res))

            if write_iter:
                mechanical_system.write_timestep(t + n_iter*0.001, u)

            # Exit, if niter too large
            if (n_iter >= n_max_iter) and conv_abort:
                u_output = np.array(u_output).T
                print(abort_statement)
                t_clock_2 = time.time()
                print('Time for static solution: ' +
                      '{0:6.3f} seconds'.format(t_clock_2 - t_clock_1))
                return u_output

        if save:
            mechanical_system.write_timestep(t, u)
        u_output.append(u.copy())
        # export iteration infos if wanted
        if track_niter:
            iteration_info.append((t, n_iter, abs_res))

    # glue the array of the iterations on the mechanical system
    mechanical_system.iteration_info = np.array(iteration_info)
    u_output = np.array(u_output).T
    t_clock_2 = time.time()
    print('Time for solving nonlinear displacements: {0:6.3f} seconds'.format(
        t_clock_2 - t_clock_1))
    return u_output


def solve_linear_statics(
        mechanical_system,
        t=1):
    '''
    Solves the linear static problem of the mechanical system.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be linearized at zero displacement and solved.
    t : float
        Time for evaluation of external force in MechanicalSystem.

    Returns
    -------
    q : ndaray
        Static solution displacement field.

    '''
    # prepare mechanical_system
    mechanical_system.clear_timesteps()

    print('Assembling external force and stiffness')
    K = mechanical_system.K(u=None, t=t)
    f_ext = mechanical_system.f_ext(u=None, du=None, t=t)
    mechanical_system.write_timestep(0, 0*f_ext) # write undeformed state

    print('Start solving linear static problem')
    q = solve_sparse(K, f_ext)
    mechanical_system.write_timestep(t, q) # write deformed state
    print('Static problem solved')
    return q


def solve_linear_statics_state_space(
        mechanical_system,
        t=1):
    '''
    Solves the linear static problem of the state-space system.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystemStateSpace
        State space system to be linearized at zero displacement and solved.
    t : float
        Time for evaluation of external force in MechanicalSystem.

    Returns
    -------
    x : ndaray
        Static solution state field.

    '''
    # prepare mechanical_system
    mechanical_system.clear_timesteps()

    print('Assembling external force and system matrix')
    A = mechanical_system.A(x=None, t=t)
    F_ext = mechanical_system.F_ext(x=None, t=t)
    mechanical_system.write_timestep(0, 0*F_ext) # write undeformed state

    print('Start solving linear static problem')
    x = solve_sparse(A, F_ext)
    mechanical_system.write_timestep(t, x) # write deformed state
    print('Static problem solved')
    return x

