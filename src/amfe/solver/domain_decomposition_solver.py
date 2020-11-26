#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Solver-module to solve problems, which are decomposed into subdomains.
"""

import numpy as np

try:
    from pyfeti import SerialFETIsolver
    use_pyfeti = True
except:
    use_pyfeti = False


__all__ = [
    'FETISolver',
]


class DomainDecompositionBase:
    def __init__(self):
        pass

    def solve(self, mech_systems_dict, connectors):
        pass


class FETISolver(DomainDecompositionBase):
    """
    Wrapper for FETI-solvers, provided by the PYFETI package
    """
    def __init__(self):
        super().__init__()

    def solve(self, mech_systems_dict, connectors):
        """
        Solves a non-overlapping decomposed problem.
        
        Parameters
        ----------
        mech_systems_dict : dict
            Mechanical-System Translators of each substructure
            
        connectors : dict
            Connection-matrices for the interfaces between substructures
            
        Returns
        -------
        q_dict : dict
            solutions of decomposed system for each substructure
        """
        if use_pyfeti:
            K_dict, B_dict, f_dict = self._create_K_B_f_dict(mech_systems_dict, connectors)

            fetisolver = SerialFETIsolver(K_dict, B_dict, f_dict)
            solution = fetisolver.solve()
            q_dict = solution.u_dict

            return q_dict
        else:
            raise ValueError('Could not import PYFETI-library. Please install it, or use a different solver.')

    def _create_K_B_f_dict(self, mech_systems_dict, connectors_dict):
        K_dict = dict()
        B_dict = dict()
        f_dict = dict()

        for i_system, mech_system in mech_systems_dict.items():
            u = np.zeros((mech_system.dimension, 1))
            subs_key = int(i_system)
            K_dict[subs_key] = mech_system.K(u, u, 0)
            f_dict[subs_key] = mech_system.f_ext(u, u, 0)
            B_local = dict()
            for key in connectors_dict.keys():
                if int(key[1]) == i_system:
                    local_key = (int(key[1]), int(key[0]))
                    B_local[local_key] = connectors_dict[key]

            B_dict[subs_key] = B_local

        return K_dict, B_dict, f_dict

