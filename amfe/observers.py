# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Observers module of amfe.

It handles observers that are called if some property that is updated effects other objects that
are related to the caller.
"""

import abc

__all__ = [
          ]


class Observer(abc.ABC):
    '''
    Abstract super class for all observers of the mechanical system.
    
    The tasks of the observer are:
    ------------------------------
    
    Implement an update interface for all objects that need to be informed about changes of the subject
    that calls the observer
    '''

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def update(self):
        pass
