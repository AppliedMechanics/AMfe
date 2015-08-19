# -*- coding: utf-8 -*-

'''
Test intended to fix the errors in an element formulation.
Helped to fix some issues as the tangential stiffness matrix can be
crosschecked by a finite difference approximation.

'''


import numpy as np
import scipy as sp
import time

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
    jac = np.zeros((ndof, ndof))
    h = np.sqrt(np.finfo(float).eps)
    f = func(X, u)
    for i in range(ndof):
        u_tmp = u.copy()
        u_tmp[i] += h
        f_tmp = func(X, u_tmp)
        jac[:,i] = (f_tmp - f) / h
    return jac


# This is exactly the element in Felippa's notes
Tri3 = False
Tri6 = True
Quad4 = False


if Tri3:
    x = np.array([0,0,3,1,2,2.])
    u = np.array([0,0,-0.5,0,0,0.])
    # u *= 0
    element_tri3 = amfe.Tri3(E_modul=60, poisson_ratio=1/4)
    my_element = element_tri3
elif Tri6:
    x = np.array([0,0, 3,1, 2,2, 1.5,0.5, 2.5,1.5, 1,1])
    u = np.array([0,0, -0.5,0, 0,0, -0.25,0, -0.25,0, 0,0])
    element_tri6 = amfe.Tri6(E_modul=60, poisson_ratio=1/4)
    # u *= 0
    my_element = element_tri6
elif Quad4:
    x = np.array([0,0,1,0,1,1,0,1.])
    u = np.array([0,0,0,0,0,0,0,0.])
    element_quad4 = amfe.Quad4(E_modul=1, poisson_ratio=0)
    my_element = element_quad4
else: print('Kein Element ausgewählt')


t1 = time.time()
K = my_element.k_int(x, u)
t2 = time.time() - t1
print('Benoetigte Zeit zum Aufstellen der Elementsteifigkeitsmatrix: {0}'.format(t2))

if not Quad4:
    my_element.f_int(x, u)
    el = my_element
    K_finite_diff = jacobian(el.f_int, x, u)

#print('Difference between analytical and approximated tangential stiffness matrix')
#print(K - K_finite_diff)

    print('Maximum absolute deviation:', np.max(abs(K - K_finite_diff)))
    print('Maximum relative deviation:', np.max(abs(K - K_finite_diff))/np.max(abs(K)))


M = my_element.m_int(x, u)
lambda_m = sp.linalg.eigvalsh(M)
lambda_k = sp.linalg.eigvalsh(K)



# Quad4:
rand = sp.rand(8)*0.5
x = np.array([1.,1,2,1,2,2,1,2])
u = np.array([0., 0, 0, 0, 0, 0, 0, 0])
#distort the Element
x += rand


my_quad_element = amfe.Quad4(E_modul=60, poisson_ratio=1/4, density=1.)
K = my_quad_element.k_int(x, u)
M = my_quad_element.m_int(x, u)

fg_quad_element = amfe.Quad4_FG(E_modul=60, poisson_ratio=1/4, density=1.)
K_fg = fg_quad_element.k_int(x, u)
M_fg = fg_quad_element.m_int(x, u)


# Test of the mass matrices in the different procedures
print('Massen-Matrix nach JR')
print(M/M[0,0]*4)
print('Massen-Matrix nach FG')
print(M_fg/M_fg[0,0]*4)

print('Vergleich der gesamten Masse des Elements')
print('Gesamtmasse JR:', np.sum(M), 'Gesamtmasse FG:', np.sum(M_fg))


import matplotlib.pyplot as plt
x_plot = x.reshape(-1, 2)

plt.plot(x_plot[:,0], x_plot[:,1])
#el = my_quad_element
#K_finite_diff = jacobian(el.f_int, x, u)


#
##Quad8:
#x = np.array([1.,1,2,1,2,2,1,2, 1.5, 1, 2, 1.5, 1.5, 2, 1, 1.5])
#u = np.array([0., 0, 0, 0, 0, 0, 0, 0, 0., 0, 0, 0, 0, 0, 0, 0])
#my_quad_element = amfe.Quad8(E_modul=60, poisson_ratio=1/4, density=1.)
#K, f = my_quad_element.k_and_f_int(x, u)
#M = my_quad_element.m_int(x, u)
#
#el = my_quad_element
#K_finite_diff = jacobian(el.f_int, x, u)

#%%
#
#Tetra4
x = np.array([0, 0, 0,  1, 0, 0,  0, 1, 0,  0, 0, 1.])
u = np.array([0, 0, 0,  1, 0, 0,  0, 0, 0,  0, 0, 0.])
my_tetra_element = amfe.Tetra4(E_modul=60, poisson_ratio=1/4, density=1.)
#x += sp.rand(12)*0.5
#u += sp.rand(4*3)*0.5

K, f = my_tetra_element.k_and_f_int(x, u)
M = my_tetra_element.m_int(x, u)
el = my_tetra_element
u_tmp = u.copy()
u_tmp[0] += 0.2
el.f_int(x, u_tmp)
K_finite_diff = jacobian(el.f_int, x, u)

#%%

if False:
    # Routine for testing the compute_B_matrix_routine
    # Try it the hard way:
    ndim = 2
    B_tilde = sp.rand(ndim,4)
    F = sp.rand(ndim, ndim)
    S_v = sp.rand(ndim*(ndim+1)/2)

    if ndim == 2:
        S = np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
    else:
        S = np.array([[S_v[0], S_v[5], S_v[4]],
                      [S_v[5], S_v[1], S_v[3]],
                      [S_v[4], S_v[3], S_v[2]]])

    B = amfe.compute_B_matrix(B_tilde, F)
    res1 = B.T.dot(S_v)
    res2 = B_tilde.T.dot(S.dot(F.T))
    print(res1 - res2.reshape(-1))

#%%


