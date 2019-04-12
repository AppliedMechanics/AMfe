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
    
    def __init__(self, leafpaths, arg_components=None):
        super().__init__()
        
        self.components = []
        
        # Stores the addresses of leaves in the composite-tree-structure
        self.leaf_paths = leafpaths

        if arg_components:
            if not isinstance(arg_components, Iterable):
                arg_components = [arg_components]
            for component in arg_components:
                self.add_component(component)
                
        self.component_connector = ComponentConnector()

    @property
    def no_of_components(self):
        return len(self.components)

    def add_component(self, new_component):
        """
        Adds component to composite as child

        Parameters
        ----------
        new_component : ComponentBase
            iterable containing component objects that shall be added as children to composite

        Returns
        -------
        iloc : int
            local index where the component is added in components property
        """
        self.components.append(new_component)
        return self.no_of_components-1
        
    def delete_component(self, target_component_id):
        """
        Deletes a local child component by indexlocation. Take care of connections to this component and update connections after deletions!

        Parameters
        ----------
        target_component_id : int
            local index location of child component to delete

        Returns
        -------
        None
        """
        del(self.components[target_component_id])
        
    def update_tree(self, leaf_paths):
        """
        Updates leaf path reference in case of merging trees

        (This step is necessary if child components contain composites that have an old leaf path reference)

        Parameters
        ----------
        leaf_paths : LeafPaths
            LeafPaths object the composite shall be updated with

        Returns
        -------
        None
        """
        self.leaf_paths = leaf_paths

        for component in self.components:
            if isinstance(component, ComponentComposite):
                component.update_tree(leaf_paths)

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
        for slave_id, slave_comp in enumerate(self.components):
            for master_id, master_comp in enumerate(self.components): 
                if master_id is not slave_id:
                    self.component_connector.apply_compatibility_constraint(master_id, master_comp, slave_id, slave_comp)


