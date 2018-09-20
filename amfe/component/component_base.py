# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#


class ComponentBase:
    
    def __init__(self, *args, **kwargs):
        self.mesh = None
        self.assembly = None
        self.constraint = None
        self.elements = None
        
    def partition(self, num_components, element_id_sets):
        """
        TODO: Implement subroutines to split up component
        """
