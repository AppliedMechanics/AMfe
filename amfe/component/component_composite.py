# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
from amfe.component.component_base import ComponentBase


class ComponentComposite(ComponentBase):
    """
    Class which handles child-components and child-ComponentComposites and acts as an interface to foreign clients
    """
    
    TYPE = 'ComponentComposite'
    
    def __init__(self, arg_component=()):
        super().__init__()
        self.components = arg_component
    
    @property
    def no_of_components(self):
        return len(self.components)
        
    def add_component(self, new_component):
        self.components.append(new_component)
            
    def delete_component(self, id_target_component):
        """
        TODO: Check connections (e.g. constraints) to other components and delete them first
        """
        del(self.components[id_target_component])

    def partition(self, id_target_component, num_components, element_id_sets):
        """
        TODO: adapt function input to METIS partitioning output... returning sets of elements for each new component might be most convenient
        """
        pass
        # self.components[id_target_component] = ComponentComposite(self.components[id_target_component].partition(num_components, element_id_sets))

    def get_mat(self, matrix_type="K", u=None, t=0, component_id=None):
        
        if component_id is None:
            return None
        else:
            self._test_input(matrix_type, self.components[component_id].VALID_GET_MAT_NAMES)
            
            func = getattr(self.components[component_id], matrix_type)
            mat = func(u, t)

            return mat
    
    def _test_input(self, input_to_test, valid_input):
        try:
            return valid_input.index(input_to_test)
        except AttributeError as error:
            print('{} not a valid input. Please try one of the following instead: '.format(input_to_test))
            print(valid_input)
