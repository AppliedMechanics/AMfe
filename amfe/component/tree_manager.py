#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
from collections.abc import Iterable
from amfe.component.component_composite import ComponentComposite
from copy import deepcopy


class TreeBuilder:
    
    def __init__(self):
        self.leaf_paths = LeafPaths()
        self.root_composite = ComponentComposite(self.leaf_paths)
        
    def add(self, new_components, target_path=None):
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
        for comp_id, component in enumerate(new_components):
            # add components as children to the target composite
            iloc = target_composite.add_component(component)
            # iloc is local child index where the new component is added in target_composite
            if isinstance(component, ComponentComposite):
                path = target_path
                # append the new index of component in children list of the composite to the path
                path.append(iloc)
                self.leaf_paths.merge_leafpaths(path, component.leaf_paths)
                target_composite.update_tree(self.leaf_paths)
            else:
                # if component is not a composite, just add this component as a leaf
                self.leaf_paths.add_leaf(target_path, iloc)
        
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
        component_id : int
            local index of the component within the parent composite

        Returns
        -------
        None
        """
        path = target_path + [component_id]
        self.leaves[self.max_leaf_id+1] = path
            
    def merge_leafpaths(self, target_path, new_leafpaths):
        """
        Merge a LeafPaths tree into the existing one (self) at given target_path

        Parameters
        ----------
        target_path : list
            list containing the path where the subtree is inserted
        new_leafpaths : LeafPaths
            LeafPaths object containing the LeafPaths tree information of the inserted tree

        Returns
        -------
        None
        """
        for leaf in new_leafpaths.leaves:
            path = target_path + new_leafpaths.leaves[leaf]
            self.leaves[self.max_leaf_id+1] = path
            
    def delete_leaf_by_id(self, leaf_id):
        path = self.get_composite_path(leaf_id)
        del(self.leaves[leaf_id])
        self._update_paths_after_deletion(path)
        print('Leaf ', leaf_id, ' deleted.')
        
    def delete_leafs_by_path(self, path):
        for leaf_id in self.get_leafids_from_path(path):     
            del(self.leaves[leaf_id])
        self._update_paths_after_deletion(path[:-1])
        print('Leaf ', leaf_id, ' deleted.')
        
        
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
    
    def _update_paths_after_deletion(self, subpath):
        subdict = self._get_sub_dictionary_by_subpath(subpath)
        subdict = self._renumber_subpaths(subdict)
        self._update_leaves_with_subdict(subpath, subdict)
                    
    def _get_sub_dictionary_by_subpath(self, subpath):
        sub_dict = dict()
        composite_layer = len(subpath)
        for leaf_id in self.leaves:
            path = self.leaves[leaf_id]
            if composite_layer==0 or path[:composite_layer] == subpath:
                sub_dict[leaf_id] = path[composite_layer:]
        return sub_dict
    
    def _renumber_subpaths(self, subdict):
        old_id = 0
        new_id = 0
        for sub_id in subdict:
            path = subdict[sub_id]
            if path[0] > old_id and sub_id is not list(subdict.keys())[0]:
                new_id += 1
            old_id = path[0]
            path[0] = new_id
            
            subdict[sub_id] = path
        
        return subdict
    
    def _update_leaves_with_subdict(self, subpath, subdict):
        for sub_id in subdict:
            self.leaves[sub_id] = subpath + subdict[sub_id]
