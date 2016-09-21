# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:47:48 2016

@author: rutzmoser
"""

import sympy as sy
import numpy as np

xi, eta, zeta = sy.symbols('xi, eta, zeta')

xi_vec = sy.Matrix([xi, eta, zeta])

#%% Build shape functions automatically


def build_hex_8_shape_fcts(point_coords):
    no_of_nodes = point_coords.shape[0]
    N = sy.zeros(no_of_nodes,1)
    for i in range(no_of_nodes):
        xi_i, eta_i, zeta_i = point_coords[i,:]
        N[i,0] = (1+xi_i*xi)*(1+eta_i*eta)*(1+zeta_i*zeta)/8
    return N

def build_hex_20_shape_fcts(point_coords):
    no_of_nodes = point_coords.shape[0]
    N = sy.zeros(no_of_nodes,1)
    for i in range(no_of_nodes):
        xi_i, eta_i, zeta_i = point_coords[i,:]
        if xi_i and eta_i and zeta_i:
            N[i,0] = (1+xi_i*xi)*(1+eta_i*eta)*(1+zeta_i*zeta) * \
                     (xi_i*xi + eta_i*eta + zeta_i*zeta -2)/8
        else:
            N[i,0] = sy.Rational(2/8)
            for loc, coord in ((xi_i, xi), (eta_i, eta), (zeta_i, zeta)):
                if loc:
                    N[i,0] *= (1 + loc*coord)
                else:
                    N[i,0] *= (1 - coord**2)
    return N

def build_hex_27_shape_fcts(point_coords):
    no_of_nodes = point_coords.shape[0]
    N = sy.zeros(no_of_nodes,1)
    for i in range(no_of_nodes):
        xi_i, eta_i, zeta_i = point_coords[i,:]
        N[i,0] = 1
        for loc, coord in ((xi_i, xi), (eta_i, eta), (zeta_i, zeta)):
            if loc == -1:
                N[i,0] *= coord*(coord-1)/2
            elif loc == 0:
                N[i,0] *= coord*(coord+1)/2
            elif loc == 1:
                N[i,0] *= 1 - coord**2

    return N


def compute_stress_export(N_gauss, ele_point_coords_in_gauss_view):
    no_of_gauss_points = N_g.shape[0]
    no_of_nodes = ele_point_coords_in_gauss_view.shape[0]
    stress_export = sy.zeros(no_of_nodes, no_of_gauss_points)
    for i in range(no_of_gauss_points):
        for j in range(no_of_nodes):
            xi_node, eta_node, zeta_node = ele_point_coords_in_gauss_view[j,:]
            stress_export[j,i] = N_gauss[i].subs([(xi, xi_node), (eta, eta_node),
                                           (zeta, zeta_node)])
    return stress_export


#%% The nodal numbering
#
#
#       v
#3----------2            3----10----2           3----13----2
#|\     ^   |\           |\         |\          |\         |\
#| \    |   | \          | 19       | 18        |15    24  | 14
#|  \   |   |  \        11  \       9  \        9  \ 20    11 \
#|   7------+---6        |   7----14+---6       |   7----19+---6
#|   |  +-- |-- | -> u   |   |      |   |       |22 |  26  | 23|
#0---+---\--1   |        0---+-8----1   |       0---+-8----1   |
# \  |    \  \  |         \  15      \  13       \ 17    25 \  18
#  \ |     \  \ |         16 |        17|        10 |  21    12|
#   \|      w  \|           \|         \|          \|         \|
#    4----------5            4----12----5           4----16----5

#%% swap array for Hexa20 and ParaView
swap = np.array([0,1,2,3,4,5,6,7,8,11,13,9,16,18,19,17,10,12,14,15])

#%% Shape functions for Hexa8

N = sy.Matrix([(1-xi)*(1-eta)*(1-zeta),
               (1+xi)*(1-eta)*(1-zeta),
               (1+xi)*(1+eta)*(1-zeta),
               (1-xi)*(1+eta)*(1-zeta),
               (1-xi)*(1-eta)*(1+zeta),
               (1+xi)*(1-eta)*(1+zeta),
               (1+xi)*(1+eta)*(1+zeta),
               (1-xi)*(1+eta)*(1+zeta)])/8

point_coords = sy.Matrix([(-1, -1, -1),
                                   ( 1, -1, -1),
                                   ( 1,  1, -1),
                                   (-1,  1, -1),
                                   (-1, -1,  1),
                                   ( 1, -1,  1),
                                   ( 1,  1,  1),
                                   (-1,  1,  1)])

N_new = build_hex_8_shape_fcts(point_coords)
print(N_new - N)
#%% Hexa20

point_coords = sy.Matrix([[-1, -1, -1],
                          [ 1, -1, -1],
[ 1,  1, -1],
[-1,  1, -1],
[-1, -1,  1],
[ 1, -1,  1],
[ 1,  1,  1],
[-1,  1,  1],

[ 0, -1, -1], #8
[ 1,  0, -1], #9
[ 0,  1, -1], #10
[-1,  0, -1], #11

[ 0, -1,  1], #12
[ 1,  0,  1], #13
[ 0,  1,  1], #14
[-1,  0,  1], #15

[-1, -1,  0], #16
[ 1, -1,  0], #17
[ 1,  1,  0], #18
[-1,  1,  0], #19
])

N = build_hex_20_shape_fcts(point_coords)


#%% Hexa27

point_coords = sy.Matrix([[-1, -1, -1],
                          [ 1, -1, -1],
[ 1,  1, -1],
[-1,  1, -1],
[-1, -1,  1],
[ 1, -1,  1],
[ 1,  1,  1],
[-1,  1,  1],

[ 0, -1, -1],
[-1,  0, -1],
[-1, -1,  0],

[ 1,  0, -1], #11
[ 1, -1,  0], #12
[ 0,  1, -1], #13

[ 1,  1,  0], #14
[-1,  1,  0], #15
[ 0, -1,  1], #16

[-1,  0,  1], #17
[ 1,  0,  1], #18
[ 0,  1,  1], #19

[ 0,  0, -1], #20
[ 0, -1,  0], #21
[-1,  0,  0], #22

[ 0,  0,  1], #23
[ 0,  1,  0], #24
[ 1,  0,  0], #25

[ 0,  0,  0], #26

])

N = build_hex_27_shape_fcts(point_coords)

#%% Test the shape functions
def test_N(xi_val, eta_val, zeta_val):
    return N.subs([(xi, xi_val), (eta, eta_val), (zeta, zeta_val)])

for i in range(100):
    xi_val, eta_val, zeta_val = np.random.rand(3)
    sum_val = sum(test_N(xi_val, eta_val, zeta_val))
    # print(sum_val)
    np.testing.assert_almost_equal(sum_val, 1.0)

#%%

#%% Stress export for Hexa8

a = 1
gauss_points = sy.Matrix([(-a,  a,  a, 1),
                ( a,  a,  a, 1),
                (-a, -a,  a, 1),
                ( a, -a,  a, 1),
                (-a,  a, -a, 1),
                ( a,  a, -a, 1),
                (-a, -a, -a, 1),
                ( a, -a, -a, 1)])

N_g = build_hex_8_shape_fcts(gauss_points[:,:3])

print('The point coords have to fit to the Hexa8 element')
print(point_coords)
ele_point_coords_in_gauss_view = point_coords * np.sqrt(3)
stress_export = compute_stress_export(N_g, ele_point_coords_in_gauss_view)

#%% Stress export for Hexa20

a = np.sqrt(3/5)
a = 1
wa = 5/9
w0 = 8/9
gauss_points = sy.Matrix([
                     (-a, -a, -a, wa*wa*wa),
                     ( 0, -a, -a, w0*wa*wa),
                     ( a, -a, -a, wa*wa*wa),
                     (-a,  0, -a, wa*w0*wa),
                     ( 0,  0, -a, w0*w0*wa),
                     ( a,  0, -a, wa*w0*wa),
                     (-a,  a, -a, wa*wa*wa),
                     ( 0,  a, -a, w0*wa*wa),
                     ( a,  a, -a, wa*wa*wa),
                     (-a, -a,  0, wa*wa*w0),
                     ( 0, -a,  0, w0*wa*w0),
                     ( a, -a,  0, wa*wa*w0),
                     (-a,  0,  0, wa*w0*w0),
                     ( 0,  0,  0, w0*w0*w0),
                     ( a,  0,  0, wa*w0*w0),
                     (-a,  a,  0, wa*wa*w0),
                     ( 0,  a,  0, w0*wa*w0),
                     ( a,  a,  0, wa*wa*w0),
                     (-a, -a,  a, wa*wa*wa),
                     ( 0, -a,  a, w0*wa*wa),
                     ( a, -a,  a, wa*wa*wa),
                     (-a,  0,  a, wa*w0*wa),
                     ( 0,  0,  a, w0*w0*wa),
                     ( a,  0,  a, wa*w0*wa),
                     (-a,  a,  a, wa*wa*wa),
                     ( 0,  a,  a, w0*wa*wa),
                     ( a,  a,  a, wa*wa*wa),])

N_g = build_hex_20_shape_fcts(gauss_points[:,:3])

print('The point coords have to fit to the Hexa20 element')
print(point_coords)
ele_point_coords_in_gauss_view = point_coords * sy.sqrt(sy.Rational(5)/3)
stress_export = compute_stress_export(N_g, ele_point_coords_in_gauss_view)

#%% simplification of the stress export stuff... 

b = 13*np.sqrt(15)/36 + 17/12
c = (4 + np.sqrt(15))/9
d = (1 + np.sqrt(15))/36
e = (3 + np.sqrt(15))/27
f = 1/9
g = (1 - np.sqrt(15))/36
h = -2/27
i = (3 - np.sqrt(15))/27
j = -13*np.sqrt(15)/36 + 17/12
k = (-4 + np.sqrt(15))/9
l = (3 + np.sqrt(15))/18
m = np.sqrt(15)/6 + 2/3
n = 3/18
p = (- 3 + np.sqrt(15))/18
q = (4 - np.sqrt(15))/6

#%%
dN_dxi = N.jacobian(xi_vec)
