# -*- coding: utf-8 -*-

'''
Test intended to fix the errors in an element formulation.
Helped to fix some issues as the tangential stiffness matrix can be
crosschecked by a finite difference approximation.

'''

import io
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
    Compute the jacobian of func with respect to u using a finite differences scheme. 
    
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
    Plot the element. 
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
    '''
    Check the stiffness matrix and the force vector on consistency. 
    '''
    if u == None:
        u = np.zeros_like(x)
    K = element.k_int(x, u)
    K_finite_diff = jacobian(element.f_int, x, u)
    print('Maximum deviation between directly integrated stiffness matrix')
    print('and stiffness matrix from finite differences:', np.max(abs(K - K_finite_diff)))
    print('Maximum value in the integrated stiffness matrix:', np.max(abs(K)))
    return K, K_finite_diff

def read_hbmat(filename):
    '''
    reads the hbmat file and returns it in the csc format. 
    
    Parameters
    ----------
    filename : string
        string of the filename
    
    Returns
    -------
    matrix : sp.sparse.csc_matrix
        matrix which is saved in harwell-boeing format
    
    Info
    ----
    Information on the Harwell Boeing format: 
    http://people.sc.fsu.edu/~jburkardt/data/hb/hb.html
    '''
    with open(filename, 'r') as infile:
        matrix_data = infile.read().splitlines()
    
    # Analsysis of further line indices
    n_total, n_indptr, n_indices, n_data, n_rhs = map(int, matrix_data[1].split())
    matrix_keys, n_rows, n_cols, _, _ = matrix_data[2].split()
    
    n_rows, n_cols = int(n_rows), int(n_cols)

    symmetric = False
    if matrix_keys[1] == 'S':
        symmetric = True

    idx_0 = 4
    if n_rhs > 0:
        idx_0 += 1

    indptr = sp.zeros(n_indptr, dtype=int)
    indices = sp.zeros(n_indices, dtype=int)
    data = sp.zeros(n_data)
    
    indptr[:] = list(map(int, matrix_data[idx_0 : idx_0 + n_indptr]))
    indices[:] = list(map(int, matrix_data[idx_0 + n_indptr : idx_0 + n_indptr + n_indices]))
    # consider the fortran convention with D instead of E in double precison floats
    data[:] = [float(x.replace('D', 'E')) for x in matrix_data[idx_0 + n_indptr + n_indices : ]]
    
    # take care of the indexing notation of fortran
    indptr -= 1
    indices -= 1

    matrix = sp.sparse.csc_matrix((data, indices, indptr), shape=(n_rows, n_cols))
    if symmetric:
        diagonal = matrix.diagonal()
        matrix = matrix + matrix.T
        matrix.setdiag(diagonal)
    return matrix

def produce_apdl(filename_mat, nodes, elements, ANSYS_element, rho, E, nu):
    '''
    Produce an apdl file which produces mass- and stiffness matrix for a single element. 
    
    Parameters
    ----------
    nodes : ndarray
        array containing the nodes of the element. The nodes are given in matrix format
    elements : ndarray
        array containing the nodes forming the elements. The array has to be a 2D array
    ANSYS_element : string
        string containing the elment key number and eventually the keyarguments for the APDL file
    rho : float
        density of the elements
    E : float
        Yong's modulus of the elements
    nu : float
        poisson ratio of the elements
        
    Returns
    -------
    None
    
    '''
    
    # check, if nodes are 2D
    if nodes.shape[-1] == 2:
        nodes3D = sp.zeros((nodes.shape[0], 3))
        nodes3D[:,:2] = nodes
    else:
        nodes3D = nodes
        
    apdl = io.StringIO() # works the same as files do
    apdl.write('finish \n/clear,start \n/PREP7\n\n')
    apdl.write('!Nodal values\n')
    for i, node in enumerate(nodes3D):
        apdl.write('N,' + str(i+1) + ',' + ','.join(str(x) for x in node) + '\n')
    
    apdl.write('\n!Material properties\n')
    apdl.write('MP,DENS,1,' + str(rho) + '\n')
    apdl.write('MP,EX,1,' + str(E) + '\n')
    apdl.write('MP,NUXY,1,' + str(nu) + '\n')
    apdl.write('MAT,1\n')
    
    apdl.write('\n!Elements\n')
    apdl.write('ET,1,' + ANSYS_element + '\n')
    apdl.write('TYPE,1\n')
    for i, element in enumerate(elements):
        apdl.write('E,' + ','.join(str(x+1) for x in element) + '\n')
    apdl.write('\n!Solver\n')
    apdl.write('/SOLU \nantype,modal \nmodopt,lanb,1 \nmxpand,1 \nsolve\n')
    apdl.write('\n!Postprocessor\n')
    apdl.write('/POST1 \n/AUX2 \nFILE,,full\n')
    apdl.write('HBMAT,' + filename_mat + '_m,ansmat,,ASCII,MASS,NO,NO\n')
    apdl.write('HBMAT,' + filename_mat + '_k,ansmat,,ASCII,STIFF,NO,NO\n\n\n')
    tmp = apdl.getvalue()
    apdl.close()
    return tmp

#%%

ansys_dict = {'Quad4' : '182', 
              'Quad8' : '183', 
              'Tri3' : '182', # the last two nodes should be the same
              'Tri6' : '183,1,,0', # plane stress
              'Tet4' : '285', 
              'Tet10' : '187'}

ansys_workdir = '/Volumes/ne89mez/ANSYS/'

#%%

print('Setting up the material. ')
material = amfe.KirchhoffMaterial(E=60, nu=1/4, rho=1, thickness=1)
#material = amfe.NeoHookean(0.1, 80, 1)
#%% 
# Test of the different elements:

# Tri3
print('''
###############################################################################
######  Testing Tri3 Element
###############################################################################
''')
x = np.array([0,0,3,1,2,2.])
x += sp.rand(6)
u = sp.rand(2*3)
plot_element(x, u, title='Tri3')
element_tri3 = amfe.Tri3(material)
force_test(element_tri3, x, u)

M = element_tri3.m_int(x, u)
K0 = element_tri3.k_int(x, np.zeros_like(x))
K = element_tri3.k_int(x, u)
print('The total mass of the element (in one direction) is', np.sum(M)/2 )

apdl_tri3 = produce_apdl('tri3', x.reshape((-1,2)), 
                          np.array([[0,1,2,2],]), ansys_dict['Tri3'], rho=1., E=60., nu=1/4)

with open(ansys_workdir + 'tri3.inp', 'w') as inpfile:
    inpfile.write(apdl_tri3)


#%%
_ = input('Run ' + ansys_workdir + 'tri3.inp with ANSYS!')

K_ans = read_hbmat(ansys_workdir + 'tri3_k.ansmat').A
M_ans = read_hbmat(ansys_workdir + 'tri3_m.ansmat').A

print('Difference between ANSYS and AMfe for tri3:\n')
print('Stiffness matrix:', np.max(abs(K_ans - K0)))
print('Mass matrix:', np.max(abs(M_ans - M)))

#%%

print('''
###############################################################################
######  Testing Tri6 Element
###############################################################################
''')
x = np.array([0,0, 3,1, 2,2, 1.5,0.5, 2.5,1.5, 1,1])
x += sp.rand(12)*0.4
u = np.array([0,0, -0.5,0, 0,0, -0.25,0, -0.25,0, 0,0])
element_tri6 = amfe.Tri6(E_modul=60, poisson_ratio=1/4, density=1., element_thickness=1.)
plot_element(x, u, title='Tri6', element_loop=(0, 3, 1, 4, 2, 5))

force_test(element_tri6, x, u)

M = element_tri6.m_int(x, u)
K0 = element_tri6.k_int(x, np.zeros_like(x))
K = element_tri6.k_int(x, u)
print('The total mass of the element (in one direction) is', np.sum(M)/2 )

apdl_tri6 = produce_apdl('tri6', x.reshape((-1,2)), 
                          np.array([[0,1,2,3,4,5],]), ansys_dict['Tri6'], rho=1., E=60., nu=1/4)

with open(ansys_workdir + 'tri6.inp', 'w') as inpfile:
    inpfile.write(apdl_tri6)

#%%
_ = input('Run ' + ansys_workdir + 'tri6.inp with ANSYS!')

K_ans = read_hbmat(ansys_workdir + 'tri6_k.ansmat').A
M_ans = read_hbmat(ansys_workdir + 'tri6_m.ansmat').A

print('Difference between ANSYS and AMfe for tri6:\n')
print('Maximum value stiffness matrix:', np.max(abs(K_ans - K0)))
print('Maximum value mass matrix:', np.max(abs(M_ans - M)))


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
x += sp.rand(8)*0.5
u = np.array([0,0,0,0,0,0,0,0.])
element_quad4 = amfe.Quad4(E_modul=60, poisson_ratio=1/4, density=1., element_thickness=1.)
plot_element(x, u, title='Quad4')

force_test(element_quad4, x, u)


M = element_quad4.m_int(x, u)
K0 = element_quad4.k_int(x, np.zeros_like(x))
print('The total mass of the element (in one direction) is', np.sum(M)/2 )


apdl_quad4 = produce_apdl('quad4', x.reshape((-1,2)), 
                          np.array([[0,1,2,3],]), ansys_dict['Quad4'], rho=1., E=60., nu=1/4)

with open(ansys_workdir + 'quad4.inp', 'w') as inpfile:
    inpfile.write(apdl_quad4)


#%%

_ = input('Run ' + ansys_workdir + 'quad4.inp with ANSYS!')
K_ans = read_hbmat(ansys_workdir + 'quad4_k.ansmat').A
M_ans = read_hbmat(ansys_workdir + 'quad4_m.ansmat').A

print('Difference between ANSYS and AMfe for quad4:\n')
print('Maximum deviations:\n Mass Matrix', np.max(abs(M_ans - M)))
print('Stiffness Matrix', np.max(abs(K0 - K_ans)))
#print('Stiffness matrix:', K_ans - K0)
#print('Mass matrix:', M_ans - M)



#%%

print('''
###############################################################################
######  Testing Quad8 Element
###############################################################################
''')

x = np.array([1.,1,2,1,2,2,1,2, 1.5, 1, 2, 1.5, 1.5, 2, 1, 1.5])
x += sp.rand(16)*0.4
u = sp.zeros(16)
element_quad8 = amfe.Quad8(E_modul=60, poisson_ratio=1/4, density=1., element_thickness=1.)
plot_element(x, u, title='Quad8', element_loop=(0,4,1,5,2,6,3,7))

K, K_finite_diff = force_test(element_quad8, x, u)

M = element_quad8.m_int(x, u)
K0 = element_quad8.k_int(x, np.zeros_like(x))

print('The total mass of the element (in one direction) is', np.sum(M)/2 )



apdl_quad8 = produce_apdl('quad8', x.reshape((-1,2)), 
                          np.array([[0,1,2,3,4,5,6,7],]), ansys_dict['Quad8'], rho=1., E=60., nu=1/4)

with open(ansys_workdir + 'quad8.inp', 'w') as inpfile:
    inpfile.write(apdl_quad8)


#%%

_ = input('Run ' + ansys_workdir + 'quad8.inp with ANSYS!')
K_ans = read_hbmat(ansys_workdir + 'quad8_k.ansmat').A
M_ans = read_hbmat(ansys_workdir + 'quad8_m.ansmat').A

print('Difference between ANSYS and AMfe for quad8:\n')
print('Maximum deviations:\n Mass Matrix', np.max(abs(M_ans - M)))
print('Stiffness Matrix', np.max(abs(K0 - K_ans)))


#%%

print('''
###############################################################################
######  Testing Tet4 Element
###############################################################################
''')

x = np.array([0, 0, 0,  1, 0, 0,  0, 1, 0,  0, 0, 1.])
u = np.array([0, 0, 0,  1, 0, 0,  0, 0, 0,  0, 0, 0.])

x += sp.rand(3*4)*0.4

element_tet4 = amfe.Tet4(E_modul=60, poisson_ratio=1/4, density=1.)
plot_element(x, u, title='Tet4', three_d=True)

K, K_finite_diff = force_test(element_tet4, x, u)

M = element_tet4.m_int(x, u)
K0 = element_tet4.k_int(x, np.zeros_like(x))

print('The total mass of the element (in one direction) is', np.sum(M)/3 )


apdl_tet4 = produce_apdl('tet4', x.reshape((-1,3)), 
                          np.array([[0,1,2,3],]), ansys_dict['Tet4'], rho=1., E=60., nu=1/4)

with open(ansys_workdir + 'tet4.inp', 'w') as inpfile:
    inpfile.write(apdl_tet4)


#%%

_ = input('Run ' + ansys_workdir + 'tet4.inp with ANSYS!')
K_ans = read_hbmat(ansys_workdir + 'tet4_k.ansmat').A
M_ans = read_hbmat(ansys_workdir + 'tet4_m.ansmat').A

# correct the stuff...
selector = np.ix_([0,1,2,4,5,6,8,9,10,12,13,14],[0,1,2,4,5,6,8,9,10,12,13,14])
K_ans = K_ans[selector]
M_ans = M_ans[selector]

print('Difference between ANSYS and AMfe for tet4:\n')
print('Maximum deviations:\n Mass Matrix', np.max(abs(M_ans - M)))
print('Stiffness Matrix', np.max(abs(K0 - K_ans)))




#%%

print('''
###############################################################################
######  Testing Tet10 Element
###############################################################################
''')

x = np.array([0, 0, 0,  1, 0, 0,  0, 1, 0,  0, 0, 1,  0.5, 0, 0,  0.5, 0.5, 0,  0, 0.5, 0,  0, 0, 0.5,  0.5,0,0.5,  0, 0.5, 0.5])

u = np.zeros(30)

element_tet10 = amfe.Tet10(E_modul=60, poisson_ratio=1/4, density=1.)
plot_element(x, u, title='Tet10', three_d=True)

K, K_finite_diff = force_test(element_tet10, x, u)

M = element_tet10.m_int(x, u)
K0 = element_tet10.k_int(x, np.zeros_like(x))

print('The total mass of the element (in one direction) is', np.sum(M)/3 )


apdl_tet10 = produce_apdl('tet10', x.reshape((-1,3)), 
                          np.array([[0,1,2,3,4,5,6,7,8,9],]), ansys_dict['Tet10'], rho=1., E=60., nu=1/4)


with open(ansys_workdir + 'tet10.inp', 'w') as inpfile:
    inpfile.write(apdl_tet10)


#%%

_ = input('Run ' + ansys_workdir + 'tet10.inp with ANSYS!')
K_ans = read_hbmat(ansys_workdir + 'tet10_k.ansmat').A
M_ans = read_hbmat(ansys_workdir + 'tet10_m.ansmat').A


print('Difference between ANSYS and AMfe for tet4:\n')
print('Maximum deviations:\n Mass Matrix', np.max(abs(M_ans - M)))
print('Stiffness Matrix', np.max(abs(K0 - K_ans)))


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


