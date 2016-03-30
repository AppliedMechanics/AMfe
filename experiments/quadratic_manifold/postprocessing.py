"""
Error analysis of the given files: 

Comparing the Error produced in the given files
"""
import os
import h5py
import numpy as np
import scipy as sp
import pandas as pd

import matplotlib.pyplot as plt


name_full = '/home/rutzmoser/Dokumente/004_AMfe/results/'+\
            'test_examples/20160330_093816_bar_2_f5E5_full.hdf5'

name_linearized = '/home/rutzmoser/Dokumente/004_AMfe/results/'+\
            'test_examples/20160330_093816_bar_2_f5E5_linearized.hdf5'


def h5_read_u(h5filename):
    '''
    Extract the displacement field of the given hdf5-file
    
    Parameters
    ---------
    filename : str
    
    Returns
    -------
    u_constr : ndarray
    u_unconstr : ndarray
    T : ndarray
        time
    
    '''
    with h5py.File(h5filename, 'r') as f:
        u_full = f['time_vals/Displacement'][:]
        T = f['time'][:]
        h5_mat = f['mesh/bmat']
        csr_list = []
        for par in ('data', 'indices', 'indptr', 'shape'):
            csr_list.append(h5_mat[par][:])

    bmat = sp.sparse.csr_matrix(tuple(csr_list[:3]), shape=tuple(csr_list[3]))
    
    # If the problem is 2D but exported to 3D, u_full has to be reduced. 
    ndof_unconstr, ndof_constr = bmat.shape    
    if ndof_unconstr == u_full.shape[0]:
        pass
    elif ndof_unconstr*3//2 == u_full.shape[0]: # problem is 2D but u_full is 3D
        mask = np.ones_like(u_full[:,0], dtype=bool)
        mask[2::3] = False
        u_full = u_full[mask, :]
    return bmat.T @ u_full, u_full, T

def rms_error(u_ref, u_test):
    '''
    Compute the Root-Mean-Square-Error of two time series
    '''
    delta = u_ref - u_test
    u_rms = np.zeros_like(delta[0,:])
    for i, du in enumerate(delta.T):
        # u_rms[i] = np.sqrt( (du @ du) / (u_ref[:,i] @ u_ref[:,i] + 1E-15) )
        u_rms[i] = np.sqrt(du @ du) 
    u_rms /= delta.shape[0]
    return u_rms

def displ_error(u_ref, u_test, dof_id):
    '''
    Return the displacement error 
    '''
    return (u_ref - u_test)[dof_id, :]


#%% Doing the postprocessing


node_id = 194
dof_id = node_id*2 + 1

experiments = ['linearized', 'full', 'qm_md', 'qm_smd', 'qm_smd_shift', 'qm_kry']

results_path = '/home/rutzmoser/Dokumente/004_AMfe/results/test_examples'
# This seems to work out...
# file_key = '20160330_093816_bar_2_f5E5_'
file_key = '20160330_171831_bar_arc_R1_h01_f2E5_'
file_key = '20160330_181504_bar_arc_R1_h01_f4E5_'

# Reading the displacement fields
u_dict = {}
for i, exp in enumerate(experiments):
    full_filename = os.path.join(results_path, file_key + exp + '.hdf5')
    if os.path.exists(full_filename):
        u_constr, u_unconstr, T = h5_read_u(full_filename)
        u_dict[exp] = u_unconstr
        # print('File\n', full_filename, '\nfound.')
    else:
        print('File\n', full_filename, '\nNot found.')

# Make the DataFrames for displacement and RMS-Error
df_disp = pd.DataFrame()
df_err = pd.DataFrame()
for exp in u_dict:
    df_disp[exp] = pd.Series(u_dict[exp][dof_id, :], index=T)

    rms = rms_error(u_dict['full'], u_dict[exp])
    df_err[exp] = pd.Series(rms, index=T)

df_err.plot(logy=True)
df_disp.plot()

#%%


