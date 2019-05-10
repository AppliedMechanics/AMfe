#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
from collections.abc import Iterable
from amfe.component.component_composite import ComponentComposite


class TreeBuilder:
    
    def __init__(self):
        self.leaf_paths = LeafPaths()
        self.root_composite = ComponentComposite()
        
    def add(self, new_component_ids, new_components, target_path=None):
        """

        Parameters
        ----------
        new_components : Component or Iterable containing Component objects

        target_path : list describing the path from root

        Returns
        -------
        None

        """
        # if no target path is given, set placement next to root element
        if not target_path:
            target_path = []  # Otherwise target_path gets a list with len(new_components)

        # make an iterable from the new_components parameter if necessary
        if not isinstance(new_components, Iterable):
            new_components = (new_components, )

        # find composite where the components shall be added
        target_composite = self.get_component_by_path(target_path)

        # iterate over new component objects
        for comp_id, component in zip(new_component_ids, new_components):
            # add components as children to the target composite
            target_composite.add_component(comp_id, component)

            if isinstance(component, ComponentComposite):
                path = target_path
                # append the new index of component in children list of the composite to the path
                path.append(comp_id)

                component_idpaths = component.get_full_component_idpaths()
                for path in component_idpaths:
                    self.leaf_paths.add_leaf(target_path, path)
            else:
                # if component is not a composite, just add this component as a leaf
                self.leaf_paths.add_leaf(target_path, [comp_id])
        
    def delete_leafs(self, leaf_ids):
        """
        Delete a component by leaf id

        Parameters
        ----------
        leaf_ids : int or list of int
            Leaf IDs

        Returns
        -------
        None
        """
        if not isinstance(leaf_ids, Iterable):
            leaf_ids = [leaf_ids]
        for leaf_id in leaf_ids:
            # Get parent composite
            target_component = self.get_component_by_path(self.leaf_paths.get_composite_path(leaf_id))
            # Delete child
            target_component.delete_component(self.leaf_paths.get_local_component_id(leaf_id))
            # Delete reference in leaf_paths
            self.leaf_paths.delete_leaf_by_id(leaf_id)
        
    def delete_component(self, target_path, component_id):
        """
        Delete a component by given composite path and local component id

        Parameters
        ----------
        target_path : list
            list describing the path to the parent composite

        component_id : int
            local component id in parent composite

        Returns
        -------
        None
        """
        # Get parent composite
        target_composite = self.get_component_by_path(target_path)
        # Delete child
        target_composite.delete_component(component_id)
        # Delete reference in leaf_paths
        self.leaf_paths.delete_leafs_by_path(target_path + [component_id])
        
    def get_component_by_path(self, path):
        """
        Get component defined by path

        Parameters
        ----------
        path : list
            list describing the path to a component

        Returns
        -------

        """
        target_composite = self.root_composite

        if len(path) > 0:
            for comp_id in path:
                target_composite = target_composite.components[comp_id]
                
        return target_composite


class LeafPaths:
    def __init__(self):
        self.leaves = dict()
        
    @property
    def no_of_leaves(self):    
        return len(self.leaves)
    
    @property
    def max_leaf_id(self):
        if self.no_of_leaves == 0:
            return self.no_of_leaves-1
        else:
            return max(self.leaves)
    
    def add_leaf(self, target_path, component_id):
        """
        Adds a leaf and its path to LeafPaths

        Parameters
        ----------
        target_path : list
            list containing the path to a composite where the leaf shall be added
        component_id : list of int
            local index of the component within the parent composite

        Returns
        -------
        None
        """
        path = target_path + component_id
        self.leaves[self.max_leaf_id+1] = path
            
    def delete_leaf_by_id(self, leaf_id):
        del(self.leaves[leaf_id])
        print('Leaf ', leaf_id, ' deleted.')
        
    def delete_leafs_by_path(self, path):
        for leaf_id in self.get_leafids_from_path(path):     
            self.delete_leaf_by_id(leaf_id)

    def get_composite_path(self, leaf_id, length = None):
        path = self.leaves[leaf_id]
        if len(path) == 0 or length == 0:
            return []
        else:
            if not length:
                return path[:-1]
            else:
                return path[:length]
    
    def get_leafids_from_path(self, path):
        searched_leafs = []
        for leaf_id in self.leaves:
            if path == self.get_composite_path(leaf_id, len(path)):
                searched_leafs.append(leaf_id)
            
        if len(searched_leafs)==0:
            print('Warning: No matching leaf found for component!')
        else:
            return searched_leafs
        
    def get_local_component_id(self, leaf_id, composite_layer = None):
        path = self.leaves[leaf_id]
        if composite_layer is None:
            return path[-1]
        else:
            return path[composite_layer]
