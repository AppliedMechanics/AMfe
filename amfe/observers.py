# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
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
    'MaterialObserver',
    'NodesObserver'
]


class Observer(abc.ABC):
    '''
    Abstract super class for all observers of the mechanical system.
    
    The tasks of the observer are:
    ------------------------------
    
    Implement an update interface for all objects that need to be informed about changes of the subject that calls the
    observer.
    '''

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def update(self):
        pass


class MaterialObserver(Observer):
    '''
    Observer class that updates mechanical system if a material has been changed.
    
    Attributes
    ----------
    mechanical_system: MechanicalSystem
        An instance of mechanical system that is updated by the material observer.
    '''

    def __init__(self, mechanical_system):
        '''
        Initializes a MaterialObserver for a mechanical system.
        
        Parameters:
        -----------
        mechanical_system: MechanicalSystem
            Mechanical system object that is updated by the observer when it is called.
        '''
        self.mechanical_system = mechanical_system

    def update(self):
        '''
        Updates the mechanical system object with new material information.
        '''
        self.mechanical_system.M(force_update=True)
        self.mechanical_system.D(force_update=True)


class NodesObserver(Observer):
    '''
    Observer class that updates mechanical system if nodal coordinates have been changed.

    Attributes
    ----------
    mechanical_system: MechanicalSystem
        An instance of mechanical system that is updated by the material observer.
    '''

    def __init__(self, mechanical_system):
        '''
        Initializes a NodesObserver for a mechanical system.

        Parameters:
        -----------
        mechanical_system: MechanicalSystem
            Mechanical system object that is updated by the observer when it is called.
        '''
        self.mechanical_system = mechanical_system

    def update(self):
        '''
        Updates the mechanical system object with new nodal information.
        '''
        self.mechanical_system.M(force_update=True)
        self.mechanical_system.D(force_update=True)
