#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 20.04.2015




@author: fgruber
'''

import os
import sys 

class Mesh():
    """
    Klasse Netz, in welcher die einzelnen import-Funktionen gesammelt werden
    
    Methods:
        import_msh(filepath): import files of type .msh
    
    Attributes:
    
    """
 
    def __init__(self):

        self.nodes = []
        self.elements = []
        self.properties = []
   
        
    def import_msh(self, filepath):
        tag_format_start   = "$MeshFormat"
        tag_format_end     = "$EndMeshFormat"
        tag_nodes_start    = "$Nodes"
        tag_nodes_end      = "$EndNodes"
        tag_elements_start = "$Elements"
        tag_elements_end   = "$EndElements"
        
        
        try:
            infile = open(filepath,  'r')
        except:
            print("Fehler beim Einlesen der Daten.")
            sys.exit(1)
        data_geometry = infile.read().splitlines() # Zeilenweises Einlesen der Geometriedaten
        infile.close()
        
        for s in data_geometry: # Auslesen der Indizes, bei denen die Formatliste, die Knotenliste und die Elementliste beginnen und enden
            if s == tag_format_start: # Start Formatliste
                i_format_start   = data_geometry.index(s) + 1
            elif s == tag_format_end: # Ende Formatliste
                i_format_end     = data_geometry.index(s)
            elif s == tag_nodes_start: # Start Knotenliste
                i_nodes_start    = data_geometry.index(s) + 2
                n_nodes          = int(data_geometry[i_nodes_start-1])
            elif s == tag_nodes_end: # Ende Knotenliste
                i_nodes_end      = data_geometry.index(s) 
            elif s == tag_elements_start: # Start Elementliste 
                i_elements_start = data_geometry.index(s) + 2
                n_elements       = int(data_geometry[i_elements_start-1])
            elif s == tag_elements_end: # Ende Elementliste 
                i_elements_end   = data_geometry.index(s)    

        
        if (i_nodes_end-i_nodes_start)!=n_nodes or (i_elements_end-i_elements_start)!= n_elements: # Pruefe auf Inkonsistenzen in den Dimensionen
            raise ValueError("Fehler beim Weiterverarbeiten der eingelesenen Daten! Dimensionen nicht konsistent!")


        # Extrahiere Daten aus dem eingelesen msh-File
        list_imported_mesh_format = data_geometry[i_format_start:i_format_end]
        list_imported_nodes = data_geometry[i_nodes_start:i_nodes_end]
        list_imported_elements = data_geometry[i_elements_start:i_elements_end]

        # Konvertiere die in den Listen gespeicherten Strings in Integer/Float
        for j in range(len(list_imported_mesh_format)):
            list_imported_mesh_format[j] = [float(x) for x in list_imported_mesh_format[j].split()]
        for j in range(len(list_imported_nodes)):
            list_imported_nodes[j] = [float(x) for x in list_imported_nodes[j].split()]
        for j in range(len(list_imported_elements)):
            list_imported_elements[j] = [int(x) for x in list_imported_elements[j].split()] 
           
        # Zeile [i] von [nodes] beinhaltet die X-, Y-, Z-Koordinate von Knoten [i+1]
        self.nodes = [list_imported_nodes[j][1:] for j in range(len(list_imported_nodes))]

        # Zeile [i] von [elements] beinhaltet die Knotennummern von Element [i+1]
        for j in range(len(list_imported_elements)):
            # Nur fuer Dreieckselemente!!!
            if list_imported_elements[j][1] == 2:
                tag = list_imported_elements[j][2]
                self.properties.append(list_imported_elements[j][3:3+tag])
                self.elements.append(list_imported_elements[j][3+tag:])              




if __name__ == "__main__":
    subfolder = "/gmsh"
    filename = "/2D_Rectangle_partition2.msh"
    filepath = os.path.dirname(os.path.abspath(__file__)) + subfolder + filename
    
    my_mesh = Mesh()
    my_mesh.import_msh(filepath)




