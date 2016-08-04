"""
Numerical Experiments Toolbox.

Example
-------

"""

__all__ = ['apply_async']

try:
    import dill
except Exception:
    print('Module dill not found. It is only used for the num_exp_toolbox.')

# Stuff for paralle execution
def run_dill_encoded(what):
    '''
    Run a dill encoded function, which can be spilled to a process.

    Parameters
    ----------
    what : dill.dump
        dumped tuple of function and arguments

    Returns
    -------
    fun(*args) : return value of the function
        The function executed with the arguments, as given in what.
    '''
    fun, args = dill.loads(what)
    return fun(*args)

def apply_async(pool, fun, args):
    '''
    Run a function asynchrounously.

    Parameters
    ----------
    pool : instance of multiprocessing.Pool
        A pool for running stuff in parallel
    fun : function
        function
    args : list
        arguments of function gathered in a list without keywords.

    Returns
    -------
    job_result : multiprocessing.Pool.apply instance
        Return value of the apply_async method of the Pool-class

    Examples
    --------

        >>> import multiprocessing
        >>> pool = multiprocessing.Pool()
        >>>
        >>> def func(a,b):
        >>>     return a + b
        >>>
        >>> job_result = amfe.apply_async(pool, func, [1, 2])
        >>> job_result.get()
        3

    Note
    ----
    The apply async stuff needs to pickle an object. As some nested functions
    used in amfe cannot be pickled, the dill module has to be used. This is
    done by the given method.

    '''
    return pool.apply_async(run_dill_encoded, (dill.dumps((fun, args)),))
