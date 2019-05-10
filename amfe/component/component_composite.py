#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

from collections.abc import Iterable
from .component_base import *
from .component_connector import *


class ComponentComposite(ComponentBase):
    """
    Class which handles child-components and child-ComponentComposites and acts as an interface to foreign clients
    """
    
    TYPE = 'ComponentComposite'
    
    def __init__(self, arg_components=None, componenet_ids=None):
        super().__init__()
        
        self.components = dict()

        if arg_components:
            if not isinstance(arg_components, Iterable):
                arg_components = [arg_components]
            if componenet_ids is None:
                for comp_idx, component in enumerate(arg_components):
                    self.add_component(comp_idx, component)
            else:
                for comp_id, component in zip(componenet_ids, arg_components):
                    self.add_component(comp_id, component)

        self.connector = ComponentConnector()

    @property
    def no_of_components(self):
        return len(self.components)

    def get_full_component_idpaths(self):
        """
        Searches the components-dictionary of this component and all lower composites for the components' keys.

        Parameters
        ----------
        None

        Returns
        -------
        component_idpaths : list
        """
        component_idpaths = []
        for comp_id, component in self.components.items():
            if isinstance(component, ComponentComposite):
                lower_component_idpaths = component.get_full_component_idpaths()
                for lower_comp_idpath in lower_component_idpaths:
                    component_idpaths.append([comp_id]+lower_comp_idpath)
            else:
                component_idpaths.append([comp_id])

        return component_idpaths

    def add_component(self, new_component_id, new_component):
        """
        Adds component to composite as child

        Parameters
        ----------
        new_component_id : int
            key of the new component in the components-dictionary

        new_component : ComponentBase
            component object that shall be added as child to composite

        Returns
        -------
        new_component_id : int
            key of the new component in the components-dictionary.
        """
        if isinstance(new_component, ComponentBase):
            if new_component_id in self.components:
                raise ValueError('Component-ID is already in use. Choose another one.')
            else:
                self.components[new_component_id] = new_component
        else:
            raise TypeError('Wrong type of component. Only instanciated subclasses of ComponentBase are allowed.')

    def replace_component(self, comp_id, new_component):
        """
        Replaces an existing component with a new component.

        Parameters
        ----------
        comp_id : int
            local id of the component, which shall be replaced. Id must be a key in the components-dictionary.

        new_component : ComponentBase
            component object that replaces the old component

        Returns
        -------
        None
        """
        self.components[comp_id] = new_component

    def delete_component(self, target_component_id):
        """
        Deletes a local child component by id. Take care of connections to this component and update
        connections after deletions!

        Parameters
        ----------
        target_component_id : int
            local index of child component to delete from property 'components'

        Returns
        -------
        None
        """
        del self.components[target_component_id]

    # To Test!
    def update_component_connections(self):
        """
        Updates all connection-matrices in the composite's ComponentConnector-module.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        for slave_id, slave_comp in self.components.items():
            for master_id, master_comp in self.components.items():
                if master_id is not slave_id:
                    self.connector.apply_compatibility_constraint(master_id, master_comp, slave_id, slave_comp)
