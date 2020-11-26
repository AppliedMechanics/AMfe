#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
from collections.abc import Iterable
from amfe.component.component_composite import ComponentComposite
from amfe.component.partitioner import PartitionedMeshComponentSeparator, PartitionerBase


class TreeBuilder:
    """
    Management-class for the tree structure, which results from the system's organisation in composites and their
    components. The TreeBuilder provides methods for adding and removing components or entire composites and a
    methodology to find certain components in the tree by the leaf-paths-tool.
    """
    def __init__(self, separator=PartitionedMeshComponentSeparator(), partitioner=PartitionerBase()):
        self.leaf_paths = LeafPaths()
        self.root_composite = ComponentComposite()
        self.separator = separator
        self.partitioner = partitioner

    def separate_partitioned_component_by_leafid(self, leaf_id):
        """
        This method separates a component, which has an already partitioned mesh, into new components with new submeshes.
        The new components are added to the same composite-object as the partitioned component.

        Parameters
        ----------
        leaf_id : int
            global id of the component

        Returns
        -------
        None
        """
        new_component_ids, new_components, dofs_map_loc2glo = self._separate_component(leaf_id)
        composite_path = self.leaf_paths.get_composite_path(leaf_id)

        self.delete_leafs(leaf_id)
        self.add(new_component_ids, new_components, composite_path)

        composite = self.get_component_by_path(composite_path)
        composite.connector.dofs_mapping_local2global = dofs_map_loc2glo
        composite.update_component_connections()

    def add(self, new_component_ids, new_components, target_path=None):
        """
        Adds a list of new components or tree of composites with components at a certain position in the tree.
        If no path for the positioning is given, it is just attached to the topmost composite.


        Parameters
        ----------
        new_component_ids : list, ndarray of int
            ids of the new components. Be careful with choosing unoccupied ids for the composite

        new_components : list of ComponentBase
            Component objects, that shall be added to the tree

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

    # TO TEST!
    def get_component_by_leafid(self, leaf_id):
        """
        Get component defined by global leaf-id

        Parameters
        ----------
        leaf_id : int
            global id of the desired component

        Returns
        -------
        target_component : ComponentBase
            desired component-object
        """
        return self.get_component_by_path(self.leaf_paths.leaves[leaf_id])

    def get_component_by_path(self, path):
        """
        Get component defined by path

        Parameters
        ----------
        path : list
            list describing the path to a component

        Returns
        -------
        target_component : ComponentBase
            desired component-object
        """
        target_component = self.root_composite

        if len(path) > 0:
            for comp_id in path:
                try:
                    target_component = target_component.components[comp_id]
                except ValueError:
                    print('No component of local id ', comp_id, ' found! Check your path.')

        return target_component

    # TO TEST!
    def _separate_component(self, leaf_id):
        target_component = self.get_component_by_leafid(leaf_id)
        return self.separator.separate_partitioned_component(target_component)


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
