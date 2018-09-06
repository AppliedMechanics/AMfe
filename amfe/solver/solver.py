#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Abstract super class of all solvers.
"""


from abc import ABC, abstractmethod

__all__ = [
    'abort_statement',
    'Solver'
]

abort_statement = '''
###############################################################################
#### The current computation has been aborted.                             ####
#### No convergence was gained within the number of given iteration steps. ####
###############################################################################
'''


class Solver(ABC):
    '''
    Abstract super class for all solvers of the mechanical system.
    '''

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def solve(self):
        pass
