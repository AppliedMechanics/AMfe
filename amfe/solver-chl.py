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

