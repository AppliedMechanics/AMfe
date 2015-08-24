# -*- coding: utf-8 -*-

'''
Test intended to fix the errors in an element formulation.
Helped to fix some issues as the tangential stiffness matrix can be
crosschecked by a finite difference approximation.

'''


import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt

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
    f = func(X, u).copy()
    for i in range(ndof):
        u_tmp = u.copy()
        u_tmp[i] += h
        f_tmp = func(X, u_tmp)
        jac[:,i] = (f_tmp - f) / h
    return jac


def plot_element(x, u=None, title=None, three_d=False, element_loop=None):
    '''

    '''
    ndim = 2
    if three_d:
        ndim = 3
    x_mat = x.reshape((-1, ndim))
    if not element_loop is None:
        indices = np.array(element_loop)
        x_mat = x_mat[indices]
    no_of_nodes = x_mat.shape[0]

    x1 = np.zeros(no_of_nodes + 1)
    x2 = np.zeros(no_of_nodes + 1)

    x1[:-1] = x_mat[:,0]
    x1[-1] = x_mat[0,0]
    x2[:-1] = x_mat[:,1]
    x2[-1] = x_mat[0,1]
    plt.fill(x1, x2, 'g-', label='undeformed', alpha=0.5)

    if not u is None:
        u_mat = u.reshape((-1, ndim))
        if not element_loop is None:
            u_mat = u_mat[indices]
        u1 = np.zeros(no_of_nodes + 1)
        u2 = np.zeros(no_of_nodes + 1)
        u1[:-1] = u_mat[:,0]
        u1[-1] = u_mat[0,0]
        u2[:-1] = u_mat[:,1]
        u2[-1] = u_mat[0,1]
        plt.fill(x1 + u1, x2+u2, 'r-', label='deformed', alpha=0.5)
        for i in range(no_of_nodes):
            plt.text(x_mat[i,0] + u_mat[i,0], x_mat[i,1] + u_mat[i,1], str(i), color='r')

    plt.gca().set_aspect('equal', adjustable='box')
    for i in range(no_of_nodes):
        plt.text(x_mat[i,0], x_mat[i,1], str(i), color='g')
    plt.xlim(np.min(x1)-1, np.max(x1)+1)
    plt.ylim(np.min(x2)-1, np.max(x2)+1)
    plt.grid(True)
    plt.legend()
    plt.title(title)


def force_test(element, x, u=None):
    if u == None:
        u = np.zeros_like(x)
    K = element.k_int(x, u)
    K_finite_diff = jacobian(element.f_int, x, u)
    print('Maximum deviation between directly integrated stiffness matrix')
    print('and stiffness matrix from finite differences:', np.max(abs(K - K_finite_diff)))
    print('Maximum value in the integrated stiffness matrix:', np.max(abs(K)))
    return K, K_finite_diff

#%%

# Test of the different elements:

# Tri3
print('''
###############################################################################
######  Testing Tri3 Element
###############################################################################
''')
x = np.array([0,0,3,1,2,2.])
u = sp.rand(2*3)
plot_element(x, u, title='Tri3')
element_tri3 = amfe.Tri3(E_modul=60, poisson_ratio=1/4, density=1., element_thickness=1.)
force_test(element_tri3, x, u)

M = element_tri3.m_int(x, u)
K0 = element_tri3.k_int(x, np.zeros_like(x))
K = element_tri3.k_int(x, u)
print('The total mass of the element (in one direction) is', np.sum(M)/2 )

##
## load references from ANSYS
##

#%%

print('''
###############################################################################
######  Testing Tri6 Element
###############################################################################
''')
x = np.array([0,0, 3,1, 2,2, 1.5,0.5, 2.5,1.5, 1,1])
u = np.array([0,0, -0.5,0, 0,0, -0.25,0, -0.25,0, 0,0])
element_tri6 = amfe.Tri6(E_modul=60, poisson_ratio=1/4, density=1., element_thickness=1.)
plot_element(x, u, title='Tri6', element_loop=(0, 3, 1, 4, 2, 5))

force_test(element_tri6, x, u)

M = element_tri6.m_int(x, u)
K0 = element_tri6.k_int(x, np.zeros_like(x))
K = element_tri6.k_int(x, u)
print('The total mass of the element (in one direction) is', np.sum(M)/2 )

##
## load references from ANSYS
##


#%%


print('''
###############################################################################
######  Testing Quad4 Element
###############################################################################
''')
x = np.array([0,0,1,0,1,1,0,1.])
u = np.array([0,0,0,0,0,0,0,0.])
element_quad4 = amfe.Quad4(E_modul=60, poisson_ratio=1/4, density=1., element_thickness=1.)
plot_element(x, u, title='Quad4')

force_test(element_quad4, x, u)


M = element_quad4.m_int(x, u)
K0 = element_quad4.k_int(x, np.zeros_like(x))
print('The total mass of the element (in one direction) is', np.sum(M)/2 )

##
## load references from ANSYS and compare
##


#%%

print('''
###############################################################################
######  Testing Quad8 Element
###############################################################################
''')

x = np.array([1.,1,2,1,2,2,1,2, 1.5, 1, 2, 1.5, 1.5, 2, 1, 1.5])
u = sp.rand(16)*0.4
element_quad8 = amfe.Quad8(E_modul=60, poisson_ratio=1/4, density=1., element_thickness=1.)
plot_element(x, u, title='Quad8', element_loop=(0,4,1,5,2,6,3,7))

K, K_finite_diff = force_test(element_quad8, x, u)

M = element_quad8.m_int(x, u)
K0 = element_quad8.k_int(x, np.zeros_like(x))

print('The total mass of the element (in one direction) is', np.sum(M)/2 )

##
## load references from ANSYS
##


#%%

print('''
###############################################################################
######  Testing Tet4 Element
###############################################################################
''')

x = np.array([0, 0, 0,  1, 0, 0,  0, 1, 0,  0, 0, 1.])
u = np.array([0, 0, 0,  1, 0, 0,  0, 0, 0,  0, 0, 0.])

element_tet4 = amfe.Tet4(E_modul=60, poisson_ratio=1/4, density=1.)
plot_element(x, u, title='Tet4', three_d=True)

K, K_finite_diff = force_test(element_tet4, x, u)

M = element_tet4.m_int(x, u)
K0 = element_tet4.k_int(x, np.zeros_like(x))

print('The total mass of the element (in one direction) is', np.sum(M)/3 )

##
## load references from ANSYS
##




#%%

#
# Some Gauss-Point stuff...
#


g2 = 1/3*np.sqrt(5 + 2*np.sqrt(10/7))
g1 = 1/3*np.sqrt(5 - 2*np.sqrt(10/7))
g0 = 0.0

w2 = (322 - 13*np.sqrt(70))/900
w1 = (322 + 13*np.sqrt(70))/900
w0 = 128/225

gauss_points = (
                (-g2, -g2, w2*w2), (-g2, -g1, w2*w1), (-g2,  g0, w2*w0), (-g2,  g1, w2*w1), (-g2,  g2, w2*w2),
                (-g1, -g2, w1*w2), (-g1, -g1, w1*w1), (-g1,  g0, w1*w0), (-g1,  g1, w1*w1), (-g1,  g2, w1*w2),
                ( g0, -g2, w0*w2), ( g0, -g1, w0*w1), ( g0,  g0, w0*w0), ( g0,  g1, w0*w1), ( g0,  g2, w0*w2),
                ( g1, -g2, w1*w2), ( g1, -g1, w1*w1), ( g1,  g0, w1*w0), ( g1,  g1, w1*w1), ( g1,  g2, w1*w2),
                ( g2, -g2, w2*w2), ( g2, -g1, w2*w1), ( g2,  g0, w2*w0), ( g2,  g1, w2*w1), ( g2,  g2, w2*w2))


#%%
#
# Tri6
#

x = np.array([0,0, 3,1, 2,2, 1.5,0.5, 2.5,1.5, 1,1])
x += sp.rand(12)*0.4
u = np.zeros(12)
u += sp.rand(12)*0.4



element_tri6 = amfe.Tri6(E_modul=60, poisson_ratio=1/4, density=1.)

M_1 = element_tri6.m_int(x, u)
K_1, _ = element_tri6.k_and_f_int(x, u)

# Integration Ordnung 5:

alpha1 = 0.0597158717
beta1 = 0.4701420641 # 1/(np.sqrt(15)-6)
w1 = 0.1323941527

alpha2 = 0.7974269853 #
beta2 = 0.1012865073 # 1/(6+np.sqrt(15))
w2 = 0.1259391805

element_tri6.gauss_points3 = ((1/3, 1/3, 1/3, 0.225),
                              (alpha1, beta1, beta1, w1), (beta1, alpha1, beta1, w1), (beta1, beta1, alpha1, w1),
                              (alpha2, beta2, beta2, w2), (beta2, alpha2, beta2, w2), (beta2, beta2, alpha2, w2))


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


