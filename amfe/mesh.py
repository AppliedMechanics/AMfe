# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:25:24 2015

@author: johannesr
"""

import numpy as np
import scipy as sp
import os
import sys


# Element mapping is described here. If a new element is implemented, the
# features for import and export should work when the followig list will be updated.
element_mapping_list = [
    # internal Name,    gmsh-Key, vtk/ParaView-Key, no_of_nodes, description
    ['Tri6',            9, 22, 6, 'Quadratic triangle / 6 node second order triangle'],
    ['Tri3',            2,  5, 3, 'Straight triangle / 3 node first order triangle'],
    ['Tri10',           21, 35, 10, 'Cubic triangle / 10 node third order triangle'],
    ['Quad4',           3, 9, 4, 'Bilinear rectangle / 4 node first order rectangle'],
    ['Quad8',           16, 23, 8, 'Biquadratic rectangle / 8 node second order rectangle'],
    ['straight_line',   1,  3, 2, 'Straight line composed of 2 nodes'],
    ['quadratic_line',  8, 21, 3, 'Quadratic edge/line composed of 3 nodes']
]

# actual set of implemented elements
element_set = {'Tri6', 'Tri3', 'Quad4', 'Quad8'}
line_set = {'straight_line', 'quadratic_line'}

#
# Starting from here everything's working automatically
#
gmsh2amfe        = dict([])
amfe2gmsh        = dict([])
amfe2vtk         = dict([])
amfe2no_of_nodes = dict([])


for element in element_mapping_list:
    gmsh2amfe.update({element[1] : element[0]})
    amfe2gmsh.update({element[0] : element[1]})
    amfe2vtk.update( {element[0] : element[2]})
    amfe2no_of_nodes.update({element[0] : element[3]})



def check_dir(*filenames):
    '''Checkt ob Verzeichnis vorliegt; falls nicht, wird Verzeichnis angelegt'''
    for filename in filenames:                              # loop on files
        if not os.path.exists(os.path.dirname(filename)):   # check if directory does not exists...
            os.makedirs(os.path.dirname(filename))          # then create directory
            print("Created directory: " + os.path.dirname(filename))


class Mesh:
    '''
    Class for handling the mesh operations.

    Interne Variablen:
    - nodes: Ist eine Liste bzw. ein numpy-Array, welches Angibt, wie die x-y-z-Koordinaten eines Knotens lauten. Es gibt keinen Zählindex.
    - elements: Ist eine Liste bzw. ein numpy-Array, welches angibt, welche Knoten zu welchem Element gehören. Es gibt keinen Zählindex.
    - no_of_element_nodes: Anzahl der Knoten pro Element
    - no_of_elements: Globale Anzahl der Elemente im System
    - no_of_nodes: Globale Anzahl der Knoten im System
    - element_dof: Anzahl der Freiheitsgrade eines Elements
    - node_dof: Freiheitsgrade pro Knoten; Sind je nach Elementformulierung bei reiner Verschiebung bei 2D-Problemen 2, bei 3D-Problemen 3 dofs; Wenn Rotationen betrachtet werden natürlich entsprechend mehr
    -
    '''

    def __init__(self,  node_dof=2):
        self.nodes               = []
        self.elements            = []
        self.elements_type       = []
        self.elements_properties = []
        # the displacements; They are stored as a list of numpy-arrays with shape (ndof, node_dof):
        self.u                   = None 
        self.timesteps           = []
        self.node_dof            = node_dof



    def _update_mesh_props(self):
        '''
        Just purely updates the elements props when the nodes and elements have changed
        '''
        # elements should be given as arrays; Thus this is not necessary
        if False:
            self.nodes = np.array(self.nodes)
            self.elements = np.array(self.elements)
        self.no_of_nodes = len(self.nodes)
        self.no_of_dofs = self.no_of_nodes*self.node_dof
        # element stuff
        self.no_of_elements = len(self.elements)
        self.no_of_element_nodes = len(self.elements[0])

    def read_nodes_from_csv(self, filename, node_dof=2, explicit_node_numbering=False):
        '''
        Imports the nodes list from a csv file.

        Parameters
        -----------
        filename : str
            name of the file containing the nodes in csv-format
        node_dof : int, optional
            number of degrees of freedom per node; for 2D-meshes this is 2,
            for 3D-meshes this is 3
        explicit_node_numbering : bool, optional
            Flag stating, if the nodes are explicitly numbered in the csv-file.
            When set to true, the first column is assumed to have the node numbers
            and is thus ignored.

        Returns
        --------
        None

        Examples
        ---------
        TODO

        '''
        self.node_dof = node_dof
        try:
            self.nodes = np.genfromtxt(filename, delimiter = ',', skip_header = 1)
        except:
            print('FEHLER beim lesen der Datei', filename, '\n Vermutlich stimmt die erwartete Dimension der Knotenfreiheitsgrade', node_dof, 'nicht mit der Dimension in der Datei zusammen.')
        # when line numbers are erased if they are content of the csv
        if explicit_node_numbering:
            self.nodes = self.nodes[:,1:]


    def read_elements_from_csv(self, filename, explicit_node_numbering=False):
        '''Imports the element list from a csv file.

        Parameters
        -----------
        filename : str
            name of the file containing the nodes in csv-format
        explicit_node_numbering : bool, optional
            Flag stating, if the nodes are explicitly numbered in the csv-file.
            When set to true, the first column is assumed to have the node numbers
            and is thus ignored.

        Returns
        --------
        None

        Examples
        ---------
        TODO

        '''
        mesh_type_dict = {3: "Tri3",
                          4: "Quad4"}


        print('Reading elements from csv...  ', end="")
        self.elements = np.genfromtxt(filename, delimiter = ',', dtype = int, skip_header = 1)
        if self.elements.ndim == 1: # Wenn nur genau ein Element vorliegt
            self.elements = np.array([self.elements])
        if explicit_node_numbering:
            self.elements = self.elements[:,1:]
        try:
            (no_of_ele, no_of_nodes_per_ele) = self.elements.shape
            mesh_type = mesh_type_dict[no_of_nodes_per_ele]
        except:
            print('FEHLER beim Einlesen der Elemente. Typ nicht vorhanden.')
            raise

        print('Element type is {0}...  '.format(mesh_type), end="")
        self.elements_type = [mesh_type for i in self.elements]
        self._update_mesh_props()

        print('Reading elements successful.')


    def import_msh(self, filename, flat_mesh=True):
        """
        Import the mesh file from gmsh.

        Parameters
        -----------
        filename : str
            file name of the msh-file
        flat_mesh : bool, optional
            flag for information whether mesh is flat (2D) or not flat (3D)

        Returns
        --------
        None

        Examples
        ---------
        TODO

        """

        # Setze die in gmsh verwendeten Tags
        tag_format_start   = "$MeshFormat"
        tag_format_end     = "$EndMeshFormat"
        tag_nodes_start    = "$Nodes"
        tag_nodes_end      = "$EndNodes"
        tag_elements_start = "$Elements"
        tag_elements_end   = "$EndElements"


        self.nodes               = []
        self.elements            = []
        self.elements_type       = []
        self.elements_properties = []

        with open(filename, 'r') as infile:
            data_geometry = infile.read().splitlines()

        # Auslesen der Indizes, bei denen die Formatliste, die Knotenliste und die Elementliste beginnen und enden
        for s in data_geometry:
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

        # Konsistenzcheck (Pruefe ob Dimensionen zusammenpassen)
        if (i_nodes_end-i_nodes_start)!=n_nodes or (i_elements_end-i_elements_start)!= n_elements: # Pruefe auf Inkonsistenzen in den Dimensionen
            raise ValueError("Fehler beim Weiterverarbeiten der eingelesenen Daten! Dimensionen nicht konsistent!")

        # Extrahiere Daten aus dem eingelesen msh-File
        list_imported_mesh_format = data_geometry[i_format_start   : i_format_end]
        list_imported_nodes       = data_geometry[i_nodes_start    : i_nodes_end]
        list_imported_elements    = data_geometry[i_elements_start : i_elements_end]

        # Konvertiere die in den Listen gespeicherten Strings in Integer/Float
        for j in range(len(list_imported_mesh_format)):
            list_imported_mesh_format[j] = [float(x) for x in list_imported_mesh_format[j].split()]
        for j in range(len(list_imported_nodes)):
            list_imported_nodes[j] = [float(x) for x in list_imported_nodes[j].split()]
        for j in range(len(list_imported_elements)):
            list_imported_elements[j] = [int(x) for x in list_imported_elements[j].split()]

        # Zeile [i] von [nodes] beinhaltet die X-, Y-, Z-Koordinate von Knoten [i+1]
        self.nodes = [list_imported_nodes[j][1:] for j in range(len(list_imported_nodes))]

        boundary_line_list = [] # The nodes of a line are stored here in a unordered way
        # Zeile [i] von [elements] beinhaltet die Knotennummern von Element [i+1]
        for element in list_imported_elements:
            gmsh_element_key = element[1]
            tag = element[2] # Tag information giving everything where the structure belongs to and so on...
            if gmsh_element_key in gmsh2amfe:
                # handling of the elements:
                if gmsh2amfe[gmsh_element_key] in element_set:
                    self.elements_properties.append(element[3:3+tag])
                    self.elements.append(element[3+tag:])
                    self.elements_type.append(gmsh2amfe[gmsh_element_key])
                # Handling of the lines as they can be needed for the boundary stuff
                if gmsh2amfe[gmsh_element_key] in line_set:
                    line_number = element[2+tag]
                    if len(boundary_line_list) < line_number: # append line if line was not called
                        boundary_line_list.append([])
                    boundary_line_list[line_number - 1].append(element[3+tag:])

        # even if the nodes are heterogeneous, it should work out...
        self.nodes = np.array(self.nodes)
        self.elements = np.array(self.elements)
        # Node handling in order to make 2D-meshes flat by removing z-coordinate:
        if flat_mesh:
            self.nodes = self.nodes[:,:-1]
            self.node_dof = 2
        else:
            self.node_dof = 3
        # Take care here!!! gmsh starts indexing with 1,
        # paraview with 0!
        self.elements = np.array(self.elements) - 1

        # cleaning up redundant nodes, which may show up in gmsh files
        # This looks littel tedious but is necessary, as the 'flying' nodes
        # have neither stiffness nor mass in the assembled structure and make
        # the handling very complicated
        # Thus the preprocessing for boundary conditions has to be done with
        # paraview
        #
        # The Idea here is to make a mapping of all used nodes and the full
        # nodes and map the elements and the nodes with this mapping dict
        # called new_old_node_mapping_dict.
        used_node_set = set(self.elements.reshape(-1))
        no_of_used_nodes = len(used_node_set)
        new_old_node_mapping_dict = dict(zip(used_node_set, np.arange(no_of_used_nodes)))
        # update indexing in the element list
        for index_1, element in enumerate(self.elements):
            for index_2, node in enumerate(element):
                self.elements[index_1, index_2] = new_old_node_mapping_dict[node]
        # update indexing in the nodes list
        self.nodes = self.nodes[list(used_node_set)]

        ########
        # Postprocessing of the line_sets
        ########
        self.boundary_line_list = []
        for set_ in boundary_line_list:
            set_ = np.array(set_).reshape(-1)
            set_ = np.array(list(set(set_))) # remove the duplicates
            set_ -= 1 # consider node indexing change of gmsh
            set_ = [new_old_node_mapping_dict[node] for node in set_ if node in new_old_node_mapping_dict]
            self.boundary_line_list.append(np.array(set_))

        self._update_mesh_props()


    def set_displacement(self, u):
        '''
        Sets a displacement to the given mesh.

        Parameters
        -----------
        u : ndarray
            displacement of the unconstrained system in voigt notation

        Returns
        --------
        None

        Examples
        ---------
        TODO
        '''
        self.timesteps.append(1)
        self.u = [np.array(u).reshape((-1, self.node_dof))]


    def set_displacement_with_time(self, u, timesteps):
        '''
        Set the displacement of the mesh with the corresponding timesteps.

        expects for the timesteps a list containing the displacement vector in any shape.

        Parameters
        -----------
        u : list of ndarrays
            list containing the displacements as ndarrays in arbitrary shape
        timesteps : ndarray
            vector containing the time corresponding to the displacements in u

        Returns
        --------
        None

        Examples
        ---------
        TODO
        '''
        self.timesteps = timesteps.copy()
        self.u = []
        for i, timestep in enumerate(self.timesteps):
            self.u.append(np.array(u[i]).reshape((-1, self.node_dof)))

    def set_nodal_variable_with_time(self, variable_array, variable_name, timesteps):
        '''
        Sets the nodal variables with the time history
        '''
        # TODO
        pass

    def set_element_variable_with_time(self, variable_array, variable_name, timesteps):
        '''
        Sets the element variables with the time history
        '''
        # TODO
        pass

    def save_mesh_for_paraview(self, filename):
        '''
        Saves the mesh and the corresponding displacements to a .pvd file and corresponding .vtu-files readable for ParaView.

        In the .pvd file the information of the timesteps are given. For every timestep a .vtu file is created where the displacement and eventually the stress is saved.

        Parameters
        -----------
        filename : str
            name of the file without file endings.

        Returns
        --------
        None

        Examples
        ---------
        >>> mymesh = Mesh()
        >>> mymesh.import_msh('../meshes/my_gmsh.msh')
        >>> mymesh.save_mesh_for_paraview('../results/gmsh/my_simulation')


        References:
        -----------
        www.vtk.org/VTK/img/file-formats.pdf

        '''
        if len(self.timesteps) == 0:
            self.u = [np.zeros((self.no_of_nodes, self.node_dof))]
            self.timesteps.append(0)

        # Make the pvd-File with the links to vtu-files
        pvd_header = '''<?xml version="1.0"?> \n <VTKFile type="Collection" version="0.1" byte_order="LittleEndian">  \n <Collection> \n '''
        pvd_footer = ''' </Collection> \n </VTKFile>'''
        pvd_line_start = '''<DataSet timestep="'''
        pvd_line_middle = '''" group="" part="0" file="'''
        pvd_line_end = '''"/>\n'''
        filename_pvd = filename + '.pvd'

        filename_head, filename_tail = os.path.split(filename)

        check_dir(filename_pvd)
        with open(filename_pvd, 'w') as savefile_pvd:
            savefile_pvd.write(pvd_header)
            for i, t in enumerate(self.timesteps):
                savefile_pvd.write(pvd_line_start + str(t) + pvd_line_middle + filename_tail + '_' + str(i).zfill(3) + '.vtu' + pvd_line_end)
            savefile_pvd.write(pvd_footer)

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
        for i, t in enumerate(self.timesteps):
            filename_vtu = filename + '_' + str(i).zfill(3) + '.vtu'
            check_dir(filename_vtu)
            with open(filename_vtu, 'w') as savefile_vtu:
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
                    savefile_vtu.write(' '.join(str(x) for x in list(j)) + endflag)
                savefile_vtu.write('\n</DataArray>\n')
                savefile_vtu.write('</Points>\n<Cells>\n')
                savefile_vtu.write('<DataArray type="Int32" Name="connectivity" format="ascii">\n')
                for j in self.elements:
                    savefile_vtu.write(' '.join(str(x) for x in list(j)) + '\n')
                savefile_vtu.write('\n</DataArray>\n')
                # Writing the offset for the elements; they are ascending by the number of dofs and have to start with the real integer
                savefile_vtu.write('<DataArray type="Int32" Name="offsets" format="ascii">\n')
                for j, el_ty in enumerate(self.elements_type):
                    savefile_vtu.write(str(amfe2no_of_nodes[el_ty]*j +amfe2no_of_nodes[el_ty]) + ' ')
                savefile_vtu.write('\n</DataArray>\n')
                savefile_vtu.write('<DataArray type="Int32" Name="types" format="ascii">\n')
                savefile_vtu.write(' '.join(str(amfe2vtk[el_ty]) for el_ty in self.elements_type)) # Elementtyp ueber Zahl gesetzt
                savefile_vtu.write('\n</DataArray>\n')
                savefile_vtu.write('</Cells> \n')
                savefile_vtu.write('<PointData Vectors="displacement">\n')
                savefile_vtu.write('<DataArray type="Float64" Name="displacement" NumberOfComponents="3" format="ascii">\n')
                # pick the i-th timestep
                for j in self.u[i]:
                    savefile_vtu.write(' '.join(str(x) for x in list(j)) + endflag)
                savefile_vtu.write('\n</DataArray>\n')
                savefile_vtu.write(vtu_footer)


class MeshGenerator:
    '''
    Klasse zum Erzeugen von zweidimensionalen Netzen, die Dreieckstruktur haben.
    Ausgabe in Netz-Files, die von der Netz-Klasse wieder eingelesen werden können

    '''

    def __init__(self, x_len, y_len, x_no_elements, y_no_elements, height = 0,
                 x_curve = False, y_curve = False, flat_mesh = True,
                 mesh_style = 'Tri', pos_x0 = 0, pos_y0 = 0):
        self.x_len = x_len
        self.y_len = y_len
        self.x_no_elements = x_no_elements
        self.y_no_elements = y_no_elements
        self.x_curve = x_curve
        self.y_curve = y_curve
        self.mesh_style = mesh_style
        self.flat_mesh = flat_mesh
        self.height = height
        self.pos_x0 = pos_x0
        self.pos_y0 = pos_y0
        self.nodes = []
        self.elements = []
        # Make mesh 3D, if it is curved in one direction
        if x_curve | y_curve:
            self.flat_mesh = False
        pass

    def _curved_mesh_get_phi_r(self, h, l):
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

        def build_tri():
            '''
            Builds a triangular mesh
            '''

            # Length of one element
            l_x = self.x_len / self.x_no_elements
            l_y = self.y_len / self.y_no_elements
            # Generating the nodes
            node_number = 0 # node_number counter; node numbers start with 0
            if self.flat_mesh == True:
                for y_counter in range(self.y_no_elements + 1):
                    for x_counter in range(self.x_no_elements + 1):
                        self.nodes.append([l_x*x_counter, l_y*y_counter])
                        node_number += 1
            else:
                # a 3d-mesh will be generated; the meshing has to be done with a little calculation in andvance
                r_OO_x = np.array([0, 0, 0])
                r_OO_y = np.array([0, 0, 0])
                if self.x_curve:
                    phi_x, r_x = self._curved_mesh_get_phi_r(self.height, self.x_len)
                    delta_phi_x = phi_x/self.x_no_elements
                    r_OO_x = np.array([0, 0, -r_x])
                if self.y_curve:
                    phi_y, r_y = self._curved_mesh_get_phi_r(self.height, self.y_len)
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
                        self.nodes.append([x for x in r_OP])
            # ELEMENTS
            # Building the elements which have to be tetrahedron
            element_number = 0 # element_number counter; element numbers start with 0
            for y_counter in range(self.y_no_elements):
                for x_counter in range(self.x_no_elements):
                    # first the lower triangulars
                    first_node  = y_counter*(self.x_no_elements + 1) + x_counter + 0
                    second_node = y_counter*(self.x_no_elements + 1) + x_counter + 1
                    third_node  = (y_counter + 1)*(self.x_no_elements + 1) + x_counter + 0
                    self.elements.append([first_node, second_node, third_node])
                    element_number += 1
                    # second the upper triangulars
                    first_node  = (y_counter + 1)*(self.x_no_elements + 1) + x_counter + 1
                    second_node = (y_counter + 1)*(self.x_no_elements + 1) + x_counter + 0
                    third_node  = y_counter*(self.x_no_elements + 1) + x_counter + 1
                    self.elements.append([first_node, second_node, third_node])
                    element_number += 1


        def build_quad4():
            '''
            Builds a rectangular mesh
            '''
            delta_x = self.x_len / self.x_no_elements
            delta_y = self.y_len / self.y_no_elements

            # nodes coordinates
            for counter_y in range(self.y_no_elements+1):
                for counter_x in range(self.x_no_elements+1):
                    self.nodes.append([delta_x*counter_x + self.pos_x0, delta_y*counter_y + self.pos_y0])

            # node assignment to quadrilateral elements
            for counter_y in range(self.y_no_elements):
                for counter_x in range(self.x_no_elements):
                    node1 = counter_x     + (counter_y - 0)*(self.x_no_elements + 1)
                    node2 = counter_x + 1 + (counter_y - 0)*(self.x_no_elements + 1)
                    node3 = counter_x + 1 + (counter_y + 1)*(self.x_no_elements + 1)
                    node4 = counter_x     + (counter_y + 1)*(self.x_no_elements + 1)
                    self.elements.append([node1, node2, node3, node4])


        mesh_type_dict = {"Tri": build_tri,
                          "Quad4": build_quad4}

        mesh_type_dict[self.mesh_style]()
        print('Mesh was generated: mesh_style =', self.mesh_style)




    def save_mesh(self, filename_nodes, filename_elements):
        '''
        Speichert das Netz ab; Funktioniert für alle Elementtypen,
        es muss also stets nur eine Liste vorhanden sein
        '''

        delimiter = ','
        newline = '\n'

        check_dir(filename_nodes, filename_elements)
        with open(filename_nodes, 'w') as savefile_nodes: # Save nodes
            # Header for file:
            if self.flat_mesh:
                header = 'x_coord' + delimiter + 'y_coord' + newline
            else:
                header = 'x_coord' + delimiter + 'y_coord' + delimiter + 'z_coord' + newline
            savefile_nodes.write(header)
            for nodes in self.nodes:
                savefile_nodes.write(delimiter.join(str(x) for x in nodes) + newline)

        with open(filename_elements, 'w') as savefile_elements: # Save elements
            # Header for the file:
            number_of_nodes = len(self.elements[0])
            if number_of_nodes == 3:
                savefile_elements.write('node_1' + delimiter + 'node_2' + delimiter + 'node_3' + newline)
            elif number_of_nodes == 4:
                savefile_elements.write('node_1' + delimiter + 'node_2' + delimiter + 'node_3' + delimiter + 'node_4' + newline)
            else:
                print("Hier lief etwas falsch. Anzahl der Knoten pro Element konnte nicht bestimmt werden.")

            for elements in self.elements:
                savefile_elements.write(delimiter.join(str(x) for x in elements) + newline)

