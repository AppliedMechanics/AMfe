# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:25:24 2015

@author: johannesr
"""

import numpy as np
import scipy as sp


class Mesh:
    '''
    Die Netz-Klasse, die für die Verarbeitung des Netzes und die zugehörigen Operationen zuständig ist
    Features
    - Import von Netzdaten aus Textdateien
    - Export von Netzdaten und Verschiebungsvektoren in Textdaten
    - Zusammenarbeit mit ParaView
    - Bereitstellung von Gather- und Assembly-Matrizen
    -

    Interne Variablen:
    - nodes: Ist eine Liste bzw. ein numpy-Array, welches Angibt, wie die x-y-z-Koordinaten eines Knotens lauten. Die erste Spalte ist der Knotenindex (Zählung beginnt bei 0)
    - elements: Ist eine Liste bzw. ein numpy-Array, welches angibt, welche Knoten zu welchem Element gehören. Die erstes Spalte ist der Elementindex (Zählung beginnt bei 0)
    - no_of_element_nodes: Anzahl der Knoten pro Element
    - no_of_elements: Globale Anzahl der Elemente im System
    - no_of_nodes: Globale Anzahl der Knoten im System
    - element_dof: Anzahl der Freiheitsgrade eines Elements
    - node_dof: Freiheitsgrade pro Knoten; Sind je nach Elementformulierung bei reiner Verschiebung bei 2D-Problemen 2, bei 3D-Problemen 3 dofs; Wenn Rotationen betrachtet werden natürlich entsprechend mehr
    -
    '''

    def __init__(self):
        self.nodes = []
        self.elements = []

    def read_nodes(self, filename, node_dof=2):
        '''
        Liest die Knotenwerte aus der Datei Filename aus
        updated interne Variablen
        '''
        self.node_dof = node_dof
        if node_dof == 2:
            dtype = (int, float, float)
        elif node_dof== 3:
            dtype = (int, float, float, float)
        else:
            raise('Dimensionen passen nicht zum Programm!')
        try:
            self.nodes = np.genfromtxt(filename, delimiter = ',', dtype = dtype,  skip_header = 1)
        except:
            print('FEHLER beim lesen der Datei', filename, '\n Vermutlich stimmt die erwartete Dimension der Knotenfreiheitsgrade', node_dof, 'nicht mit der Dimension in der Datei zusammen.')
        self.no_of_nodes = len(self.nodes)

    def read_elements(self, filename):
        '''Liest die Elementmatrizen aus'''
        self.elements = np.genfromtxt(filename, delimiter = ',', dtype = int, skip_header = 1)
        self.no_of_element_nodes = len(self.elements[0]) - 1
        self.no_of_elements = len(self.elements)

    def provide_assembly_matrix(self, no_of_element):
        '''
        returns the assembly matrix B
        '''
        self.element_dof = self.no_of_element_nodes*self.node_dof
        row_indices = np.arange(self.element_dof)
        column_indices = []
        for x in self.elements[no_of_element][1:]:
            for j in range(self.node_dof):
                column_indices.append(x)
        ones = np.ones(self.element_dof)
        B = sp.sparse.csr_matrix((ones, (row_indices, column_indices)), shape = (self.element_dof, self.no_of_nodes*self.node_dof))
        return B


    def save_mesh_for_paraview(self, filename):
        '''
        Speichert das Netz für ParaView ab. Die Idee ist, dass eine Hauptdatei mit Endung .pvd als Hauptdatei für Paraview erstellt wird und anschließend das Netz in .vtu-Dateien entsprechend den Zeitschritten abgespeichert wird.
        '''
        # Make the pvd-File with the links to vtu-files
        pvd_header = '''<?xml version="1.0"?> \n <VTKFile type="Collection" version="0.1" byte_order="LittleEndian">  \n <Collection> \n '''
        pvd_footer = ''' </Collection> \n </VTKFile>'''
        pvd_line_start = '''<DataSet timestep="0" group="" part="0" file="'''
        pvd_line_end = '''"/>\n'''
        filename_pvd = filename + '.pvd'
        savefile_pvd = open(filename_pvd, 'w')
        savefile_pvd.write(pvd_header)
        timesteps = 1
        for i in range(timesteps):
            savefile_pvd.write(pvd_line_start + filename + '_' + str(i).zfill(3) + '.vtu' + pvd_line_end)
        savefile_pvd.write(pvd_footer)
        savefile_pvd.close()

        vtu_header = '''<?xml version="1.0"?> \n
        <VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
        <UnstructuredGrid>\n'''
        vtu_footer = '''
        </PointData>
        <CellData>
        </CellData>
        </Piece>
        </UnstructuredGrid>
        </VTKFile>'''
        for i in range(timesteps):
            filename_vtu = filename + '_' + str(i).zfill(3) + '.vtu'
            savefile_vtu = open(filename_vtu, 'w')
            savefile_vtu.write(vtu_header)
            # Es muss die Anzahl der gesamten Punkte und Elemente angegeben werden
            savefile_vtu.write('<Piece NumberOfPoints="' + str(len(self.nodes)) + '" NumberOfCells="' + str(len(self.elements)) + '">\n')
            savefile_vtu.write('<Points>\n')
            savefile_vtu.write('<DataArray type="Float64" Name="Array" NumberOfComponents="3" format="ascii">\n')
            # bei Systemen mit 2 Knotenfreiheitsgraden wird die dritte 0-Komponenten noch extra durch die endflag hinzugefügt...
            if self.node_dof == 2:
                endflag = ' 0 \n'
            elif self.node_dof == 3:
                endflag = '\n'
            for j in self.nodes:
                savefile_vtu.write(' '.join(str(x) for x in list(j)[1:]) + endflag)
            savefile_vtu.write('\n</DataArray>\n')
            savefile_vtu.write('</Points>\n<Cells>\n')
            savefile_vtu.write('<DataArray type="Int32" Name="connectivity" format="ascii">\n')
            for j in self.elements:
                savefile_vtu.write(' '.join(str(x) for x in list(j)[1:]) + '\n')
            savefile_vtu.write('\n</DataArray>\n')
            # Writing the offset for the elements; they are ascending by the number of dofs and have to start with the real integer
            savefile_vtu.write('<DataArray type="Int32" Name="offsets" format="ascii">\n')
            for j in self.elements:
                savefile_vtu.write(str((j[0] + 1)*3) + ' ')
            savefile_vtu.write('\n</DataArray>\n')
            savefile_vtu.write('<DataArray type="Int32" Name="types" format="ascii">\n')
            savefile_vtu.write(' '.join('5' for x in self.elements))
            savefile_vtu.write('\n</DataArray>\n')
            savefile_vtu.write('</Cells> \n<PointData>\n')
            savefile_vtu.write('<DataArray type="Int32" Name="displacement" NumberOfComponents="2" format="ascii">\n')
            for j in self.nodes:
                savefile_vtu.write(' '.join('0' for x in list(j)[1:]) + '\n')
            savefile_vtu.write('\n</DataArray>\n')
            savefile_vtu.write(vtu_footer)
            savefile_vtu.close()
        pass

## test
#my_mesh = Mesh()
#my_mesh.read_elements('saved_elements.csv')
#my_mesh.read_nodes('saved_nodes.csv')
#my_mesh.provide_assembly_matrix(3).toarray()
#my_mesh.save_mesh_for_paraview('myfilename')

#%%

class Mesh_generator:
    '''
    Klasse zum Erzeugen von zweidimensionalen Netzen, die Dreieckstruktur haben.
    Ausgabe in Netz-Files, die von der Netz-Klasse wieder eingelesen werden können

    3d_mesh = False
    '''

    def __init__(self, x_len, y_len, x_no_elements, y_no_elements, height = 0, x_curve = False, y_curve = False, flat_mesh = True, mesh_style = 'tetra'):
        self.x_len = x_len
        self.y_len = y_len
        self.x_no_elements = x_no_elements
        self.y_no_elements = y_no_elements
        self.x_curve = x_curve
        self.y_curve = y_curve
        self.mesh_style = mesh_style
        self.flat_mesh = flat_mesh
        self.height = height
        self.nodes = []
        self.elements = []
        # Make mesh 3D, if it is curved in one direction
        if x_curve | y_curve:
            self.flat_mesh = False
        pass

    def curved_mesh_get_phi_r(self, h, l):
        '''
        wenn ein gekrümmtes Netz vorliegt:
        Bestimmung des Winkels phi und des Radiusses r aus der Höhe und der Länge

        '''
        # Abfangen, wenn Halbschale vorliegt
        if l - 2*h < 1E-7:
            phi = np.pi
        else:
            phi = 2*np.arctan(2*h*l/(l**2 - 4*h**2))
        # Checkt, wenn die Schale über pi hinaus geht:
        if phi<0:
            phi += 2*np.pi
        r = l/(2*np.sin(phi/2))
        return phi, r

    def build_mesh(self):
        '''
        Building the mesh by first producing the points, and secondly the elements
        '''
        # Length of one element
        l_x = self.x_len / self.x_no_elements
        l_y = self.y_len / self.y_no_elements
        # Generating the nodes
        node_number = 0 # node_number counter; node numbers start with 0
        if self.flat_mesh == True:
            for y_counter in range(self.y_no_elements + 1):
                for x_counter in range(self.x_no_elements + 1):
                    self.nodes.append([node_number, l_x*x_counter, l_y*y_counter])
                    node_number += 1
        else:
            # a 3d-mesh will be generated; the meshing has to be done with a little calculation in andvance
            r_OO_x = np.array([0, 0, 0])
            r_OO_y = np.array([0, 0, 0])
            if self.x_curve:
                phi_x, r_x = self.curved_mesh_get_phi_r(self.height, self.x_len)
                delta_phi_x = phi_x/self.x_no_elements
                r_OO_x = np.array([0, 0, -r_x])
            if self.y_curve:
                phi_y, r_y = self.curved_mesh_get_phi_r(self.height, self.y_len)
                delta_phi_y = phi_y/self.y_no_elements
                r_OO_y = np.array([0, 0, -r_y])
            # Einführen von Ortsvektoren, die Vektorkette zum Element geben:
            r_OP_x = np.array([0, 0, 0])
            r_OP_y = np.array([0, 0, 0])
            r_OO   = np.array([self.x_len/2, self.y_len/2, self.height])
            for y_counter in range(self.y_no_elements + 1):
                for x_counter in range(self.x_no_elements + 1):
                    if self.x_curve:
                        phi = - phi_x/2 + delta_phi_x*x_counter
                        r_OP_x = np.array([r_x*np.sin(phi), 0, r_x*np.cos(phi)])
                    else:
                        r_OP_x = np.array([- self.x_len/2 + l_x*x_counter, 0, 0])
                    if self.y_curve:
                        phi = - phi_y/2 + delta_phi_y*y_counter
                        r_OP_y = np.array([0, r_y*np.sin(phi), r_y*np.cos(phi)])
                    else:
                        r_OP_y = np.array([0, - self.y_len/2 + l_y*y_counter, 0])
                    r_OP = r_OP_x + r_OP_y + r_OO_x + r_OO_y + r_OO
                    self.nodes.append( [node_number] + [x for x in r_OP])
        # ELEMENTS
        # Building the elements which have to be tetrahedron
        element_number = 0 # element_number counter; element numbers start with 0
        for y_counter in range(self.y_no_elements):
            for x_counter in range(self.x_no_elements):
                # first the lower triangulars
                first_node  = y_counter*(self.x_no_elements + 1) + x_counter + 0
                second_node = y_counter*(self.x_no_elements + 1) + x_counter + 1
                third_node  = (y_counter + 1)*(self.x_no_elements + 1) + x_counter + 0
                self.elements.append([element_number, first_node, second_node, third_node])
                element_number += 1
                # second the upper triangulars
                first_node  = (y_counter + 1)*(self.x_no_elements + 1) + x_counter + 1
                second_node = (y_counter + 1)*(self.x_no_elements + 1) + x_counter + 0
                third_node  = y_counter*(self.x_no_elements + 1) + x_counter + 1
                self.elements.append([element_number, first_node, second_node, third_node])
                element_number += 1
        pass

    def save_mesh(self, filenname_nodes, filename_elements):
        '''
        Speichert das Netz ab; Funktioniert für alle Elementtypen,
        es muss also stets nur eine Liste vorhanden sein
        '''
        delimiter = ','
        newline = '\n'
        savefile_nodes = open(filenname_nodes, 'w')
        # Header for file:
        if self.flat_mesh:
            header = 'Node_id' + delimiter + 'x_coord' + delimiter + 'y_coord' + newline
        else:
            header = 'Node_id' + delimiter + 'x_coord' + delimiter + 'y_coord' + delimiter + 'z_coord' + newline
        savefile_nodes.write(header)
        for nodes in self.nodes:
            savefile_nodes.write(delimiter.join(str(x) for x in nodes) + newline)
        savefile_nodes.close()

        savefile_elements = open(filename_elements, 'w')
        # Header for the file:
        savefile_elements.write('Element_id' + delimiter + 'node_1' + delimiter + 'node_2' + delimiter + 'node_3' + newline)
        for elements in self.elements:
            savefile_elements.write(delimiter.join(str(x) for x in elements) + newline)
        pass


#
## Test
#my_meshgenerator = Mesh_generator(x_len=3*3, y_len=4*3, x_no_elements=3*3*3, y_no_elements=3*3*3, height = 1.5, x_curve=True, y_curve=False)
#my_meshgenerator.build_mesh()
#my_meshgenerator.save_mesh('saved_nodes.csv', 'saved_elements.csv')
#
#my_mesh = Mesh()
#my_mesh.read_elements('saved_elements.csv')
#my_mesh.read_nodes('saved_nodes.csv', node_dof=3)
#my_mesh.save_mesh_for_paraview('myfilename')