

import numpy as np

__all__ = ['MemoizeJac',
           'MemoizeStiffness',
           'MemoizeConstant',
           'MakeConstantCallable'
           ]


class MemoizeJac(object):
    """ Decorator that caches the value gradient of function each time it
    is called. """
    def __init__(self, fun):
        self.fun = fun
        self.jac = None
        self.x = None

    def __call__(self, x, *args):
        self.x = np.asarray(x).copy()
        fg = self.fun(x, *args)
        self.jac = fg[1]
        return fg[0]

    def derivative(self, x, *args):
        if self.jac is not None and np.alltrue(x == self.x):
            return self.jac
        else:
            self(x, *args)
        return self.jac


class MemoizeConstant(object):
    def __init__(self, fun):
        self.fun = fun
        self._cache = None

    def __call__(self, *args, **kwargs):
        if self._cache is None:
            self._cache = self.fun(*args, **kwargs)
        return self._cache


class MemoizeStiffness(object):
    """ Decorator that caches the value gradient of function each time it
    is called. """
    def __init__(self, K_and_f_fun):
        self.fun = K_and_f_fun
        self.jac = None
        self.q = None
        self.dq = None
        self.ddq = None
        self.t = None

    def __call__(self, q, dq, t, *args):
        self.q = np.asarray(q).copy()
        self.dq = np.asarray(dq).copy()
        self.t = t

        fg = self.fun(q, dq, t, *args)
        self.jac = fg[0]
        return fg[1]

    def derivative(self, q, dq, t, *args):
        if self.jac is not None and t == self.t and np.alltrue(q == self.q)\
                and np.alltrue(dq == self.dq):
            return self.jac
        else:
            self(q, dq, t, *args)
        return self.jac


class MakeConstantCallable(object):
    """
    Object, that stores an object at initialization and provides a caller-method of the same name, which returns the
    stored object. This is especially recommended for callback-functions for objects, that would be constructed anew at
    each call, but won't change.
    """
    def __init__(self, constant):
        self._cache = constant

    def __call__(self, *args, **kwargs):
        return self._cache