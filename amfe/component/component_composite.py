# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
from amfe.component.structural_component import StructuralComponent


class ComponentComposite:
    """
    Class which handles child-components and child-ComponentComposites and acts as an interface to foreign clients
    """
    
    name = 'ComponentComposite'
    
    def __init__(self, component_types, arg_component=()):
        
        if not arg_component:
            self.component = []*len(component_types)
            for i, component_type in enumerate(component_types):
                if component_type != 'structural':
                    print('Warning: Non-structural components currently not supported')
                else:
                    self.component[i] = StructuralComponent()
                    
        else:
            self.component = arg_component
            
        self.assembly = None
    
    def partition(self, id_target_component, num_components, element_id_sets):
        """
        TODO: adapt function input to METIS partitioning output... returning sets of elements for each new component might be most convenient
        """
        self.component[id_target_component] = ComponentComposite(self.component[id_target_component].partition(num_components, element_id_sets))

    def get_mat(self, matrix_type="K", u=None, t=0, component_id=None):
        
        if not component_id:
            mat = self.assembly.assemble(self.component)
        else:
            try:
                func = getattr(self.component[component_id], matrix_type)
                mat = func(u, t)
            except AttributeError as error:
                print('{} not found in component'.format(matrix_type))
        return mat
