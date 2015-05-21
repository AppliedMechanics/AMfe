# -*- coding: utf-8 -*-

'''
Test intended to fix the errors in an element formulation.
Helped to fix some issues as the tangential stiffness matrix can be
crosschecked by a finite difference approximation.

'''


import numpy as np
import scipy as sp

# make amfe running
import sys
sys.path.insert(0,'..')

import amfe


def jacobian(func, X, u):
    '''
    Bestimmung der Jacobimatrix auf Basis von finiten Differenzen.
    Die Funktion func(vec, X) wird nach dem Vektor vec abgeleitet, also d func / d vec an der Stelle X
    '''
    ndof = X.shape[0]
    jacobian = np.zeros((ndof, ndof))
    h = np.sqrt(np.finfo(float).eps)
    f = func(X, u)
    for i in range(ndof):
        u_tmp = u.copy()
        u_tmp[i] += h
        f_tmp = func(X, u_tmp)
        jacobian[:,i] = (f_tmp - f) / h
    return jacobian


# This is exactly the element in Felippa's notes

x = np.array([0,0,3,1,2,2.])
u = np.array([0,0,-0.5,0,0,0.])
u = np.random.rand(6)
my_element = amfe.ElementPlanar(E_modul=60, poisson_ratio=1/4)
K = my_element.k_int(x, u)
my_element.f_int(x, u)
el = my_element

K_finite_diff = jacobian(el.f_int, x, u)
K = el.k_int(x, u)

print('Difference between analytical and approximated tangential stiffness matrix')
print(K - K_finite_diff)

#%%






