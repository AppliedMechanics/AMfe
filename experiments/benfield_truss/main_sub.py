#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:22:51 2015

@author: fabian
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

# make amfe running
import sys
sys.path.insert(0, '../..')
import amfe
import plot_bar
import create_sub

# Noch Probleme bei Substruktur 2!!!, da K-Matrix 5 Null-Eigenwerte hat

np.set_printoptions(precision=7, suppress=True, linewidth=3)

l1 = 4.0

#%% Left substructure
# Check if data is already existing
path_nod = './mesh/substructures/nodes-sub1.csv'
path_ele = './mesh/substructures/elements-sub1.csv'
if not os.path.exists(path_nod) or not os.path.exists(path_ele):
    create_sub.create_substructure()
    print("Had to create Benfield truss since", path_nod, "or", path_ele,
        "did not exist\n")
    
# Building the mechanical system
my_system1 = amfe.MechanicalSystem(E_modul=1.0, crosssec=1.0, density=1.0)
my_system1.load_mesh_from_csv(path_nod, path_ele, ele_type = 'Bar2Dlumped')
M1, K1 = amfe.give_mass_and_stiffness(my_system1)

# Plot mesh of bars
pos_of_nodes1 = my_system1.node_list.reshape((-1, 1))    
plot_bar.plt_mesh(my_system1.element_list, pos_of_nodes1, plot_no_of_ele=True, 
                  plot_nodes=True, p_col='0.25', no_of_fig=1, p_title='Mesh',
                  p_col_node='r')  

dof_b = np.arange(6)
# Determine boundary and internal dofs
dof_1 = np.arange(my_system1.node_list.shape[0]*2)
nodes_b1 = np.where(my_system1.node_list[:, 0] == 5*l1)[0]
dof_b1 = np.sort(np.concatenate((nodes_b1*2, nodes_b1*2+1), axis=1))
dof_i1 = np.setdiff1d(dof_1,dof_b1)                           
B1 = sp.sparse.lil_matrix((6,dof_1.shape[0]))
B1[dof_b,dof_b1] = np.ones(6) 
B1 = B1.tocsc() 
L1 = sp.sparse.identity(6)

                  
#%% Right substructure
# Check if data is already existing
path_nod = './mesh/substructures/nodes-sub2.csv'
path_ele = './mesh/substructures/elements-sub2.csv'
if not os.path.exists(path_nod) or not os.path.exists(path_ele):
    create_sub.create_substructure()
    print("Had to create Benfield truss since", path_nod, "or", path_ele,
        "did not exist\n")                           

# Building the mechanical system
my_system2 = amfe.MechanicalSystem(E_modul=1.0, crosssec=1.0, density=1.0)
my_system2.load_mesh_from_csv(path_nod, path_ele, ele_type = 'Bar2Dlumped')
M2, K2 = amfe.give_mass_and_stiffness(my_system2)             

# Plot mesh of bars
pos_of_nodes2 = my_system2.node_list.reshape((-1, 1))    
plot_bar.plt_mesh(my_system2.element_list, pos_of_nodes2, plot_no_of_ele=True, 
                  plot_nodes=True, p_col='0.25', no_of_fig=2, p_title='Mesh',
                  p_col_node='r')  
                             

# Determine interface dofs
dof_2 = np.arange(my_system2.node_list.shape[0]*2)
nodes_b2 = np.where(my_system2.node_list[:, 0] == 5*l1)[0]
dof_b2 = np.sort(np.concatenate((nodes_b2*2, nodes_b2*2+1), axis=1))                
dof_i2 = np.setdiff1d(dof_2,dof_b2)  
B2 = sp.sparse.lil_matrix((6,dof_2.shape[0]))
B2[dof_b,dof_b2] = np.ones(6)
B2 = B2.tocsc() 
L2 = -sp.sparse.identity(6)


#%% Craig-Bampton Methode
def assemble_CBsys(substr, no_of_dof_b):
    '''
    Assemblierung von Substrukturen, welche mit Craig-Bampton Methode reduziert
    wurden. 
    '''
    
    no_of_dof = no_of_dof_b # Anzahl an Gesamt-boundary-DOF
    counter = no_of_dof_b # Counter fuer assembly
    for i, sub in enumerate(substr): # Bestimme Anzahl an DOF des reduzierten Systems
        no_of_dof += sub.no_of_modes_CB
 
    print("no_of_dof = ", no_of_dof)      
    K_CB = sp.sparse.lil_matrix((no_of_dof,no_of_dof)) # Initialize stiffness        
    M_CB = sp.sparse.lil_matrix((no_of_dof,no_of_dof)) # Initialize mass

    # Loop on substructure for assembly in reduced, global matrices
    for i, sub in enumerate(substr): 
        
        # Mass Matrix    
        M_CB[0:no_of_dof_b,0:no_of_dof_b] = M_CB[0:no_of_dof_b,0:no_of_dof_b] \
            + sub.L.T.dot(sub.M[np.ix_(sub.dof_b,sub.dof_b)] \
                            + sub.M[np.ix_(sub.dof_b,sub.dof_i)].dot(sub.psi) \
                            + sub.psi.T.dot(sub.M[np.ix_(sub.dof_i,sub.dof_b)]) \
                            + sub.psi.T.dot(sub.M[np.ix_(sub.dof_i,sub.dof_i)].dot(sub.psi))).dot(sub.L)
        M_CB[0:no_of_dof_b,counter:counter+sub.no_of_modes_CB] = \
            sub.L.T.dot(sub.M[np.ix_(sub.dof_b,sub.dof_i)] \
                        + sub.psi.T.dot(sub.M[np.ix_(sub.dof_i,sub.dof_i)])).dot(sub.phi)
        M_CB[counter:counter+sub.no_of_modes_CB,0:no_of_dof_b] = \
            M_CB[0:no_of_dof_b,counter:counter+sub.no_of_modes_CB].T
        M_CB[counter:counter+sub.no_of_modes_CB,counter:counter+sub.no_of_modes_CB] = \
            sp.sparse.identity(sub.no_of_modes_CB)

        # Stiffness matrix        
        K_CB[0:no_of_dof_b,0:no_of_dof_b] = K_CB[0:no_of_dof_b,0:no_of_dof_b] \
            + sub.L.T.dot(sub.K[np.ix_(sub.dof_b,sub.dof_b)] \
                        + sub.K[np.ix_(sub.dof_b,sub.dof_i)].dot(sub.psi)).dot(sub.L)
        K_CB[counter:counter+sub.no_of_modes_CB,counter:counter+sub.no_of_modes_CB] = \
            np.diag(sub.Om2CB) 
        counter = counter + sub.no_of_modes_CB
           
    return K_CB.tocsc(), M_CB.tocsc()        
        


class Substructure():
    '''
    Class for substructures
    '''

    def __init__(self, K=[], M=[], dof_i=[], dof_b=[]):
        self.K = K
        self.M = M
        self.dof_i = dof_i
        self.dof_b = dof_b
        
    def set_B(self, B):
        self.B = B    
        
    def set_L(self, L):
        self.L = L
       
    def CraigBampton(self, n_modes=6):
        # Eigenmodes
        self.Om2CB, self.phi = sp.sparse.linalg.eigsh(self.K[np.ix_(self.dof_i,self.dof_i)],
                            k=n_modes, M=self.M[np.ix_(self.dof_i,self.dof_i)],
                            which='SM')
        # Static mode shapes
        self.psi = sp.sparse.linalg.spsolve(self.K[np.ix_(self.dof_i,self.dof_i)], 
                                              -self.K[np.ix_(self.dof_i,self.dof_b)])   
        # Set number of modes
        self.no_of_modes_CB = n_modes            
        
    




substructure1 = Substructure(K1, M1, dof_i1, dof_b1)
substructure2 = Substructure(K2, M2, dof_i2, dof_b2)

substructure1.set_B(B1)
substructure2.set_B(B2)

substructure1.set_L(L1)
substructure2.set_L(L2)

n_modes1 = 10
n_modes2 = 10


substructure1.CraigBampton(n_modes=n_modes1)
substructure2.CraigBampton(n_modes=n_modes2)

print("Eigenfrequenzen von Substruktur 1: \n", substructure1.Om2CB)
print("Eigenfrequenzen von Substruktur 2: \n", substructure2.Om2CB)

substr_list =  [substructure1, substructure2]
K_CB, M_CB = assemble_CBsys(substr_list, no_of_dof_b = 6)


Om2glo, phi = sp.sparse.linalg.eigsh(K_CB, k=15, M=M_CB, which='SM')
print("Eigenfrequenzen des reduzierten Gesamtsystems: \n", Om2glo)




#%% Plotausgabe
pos_of_nodes1 = my_system1.node_list.reshape((-1, 1))
disp_fix1 = np.zeros((dof_1.shape[0],substructure1.no_of_modes_CB))
disp_fix1[dof_i1,:] = substructure1.phi
scale = 10
disp1 = pos_of_nodes1 + scale*disp_fix1
 
plot_bar.plt_mesh(my_system1.element_list, disp1[:,3], plot_no_of_ele=True, 
                  plot_nodes=True, p_col='0.25', no_of_fig=3, p_title='Shape',
                  p_col_node='r') 


pos_of_nodes2 = my_system2.node_list.reshape((-1, 1))
disp_fix2 = np.zeros((dof_2.shape[0],substructure2.no_of_modes_CB))
disp_fix2[dof_i2,:] = substructure2.phi
scale = 10
disp2 = pos_of_nodes2 + scale*disp_fix2
  
plot_bar.plt_mesh(my_system2.element_list, disp2[:,3], plot_no_of_ele=True, 
                  plot_nodes=True, p_col='0.25', no_of_fig=4, p_title='Shape',
                  p_col_node='r')



