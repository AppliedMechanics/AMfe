Assembly
========

The assembly module has the following tasks:

* Generate Mapping tables from local dofs to global dofs

* Assemble global matrices and vectors such as

    - mass matrix
    - tangent stiffness matrix
    - internal force vector
    - external force vector
    - stresses and strains
    



Instantiate an Assembly object
------------------------------

All this tasks can be done by using the Assembly-class.
First instantiate an object of :py:class:`Assembly-class<amfe.assembly.Assembly>`::

    >>> import amfe
    >>> my_mesh = amfe.Mesh()
    >>> # some mesh operations till mesh is setup
    >>> my_assembly = amfe.Assembly(my_mesh)
    
It is neccessary to pass a :py:class:`amfe.Mesh<amfe.mesh.Mesh>` object to the constructor.
The Pointer to that mesh is stored in the :py:attr:`mesh<amfe.assembly.Assembly.mesh>`-property.

Then it is recommended to preallocate the tangential stiffness matrix for your problem.
This is done via::

    >>> my_assembly.preallocate_csr()

The method :py:meth:`preallocate_csr()<amfe.assembly.Assembly.preallocate_csr>`
preallocates the stiffness matrix with the help of the mesh-information
from your mesh object. This matrix is stored in the :py:attr:`C_csr<amfe.assembly.Assembly.C_csr>`-property.
This method also stores a pointer to a view of the node-coordinates of the :py:attr:`Mesh.nodes<amfe.mesh.Mesh.nodes>`-property
of the assigned mesh-class. This view is a vector with all node-coordiantes concatenated.


.. note::
    
    This can take a while for large systems.




Generate mapping from local dofs to global dofs
-----------------------------------------------

The assembly has two properties which define the mapping from local degrees of
freedom of each element to the global degrees of freedom.
There is one property for domain elements (:py:attr:`element_indices<amfe.assembly.Assembly.element_indices>`)
and one for boundary elements (:py:attr:`neumann_indices<amfe.assembly.Assembly.neumann_indices>`)

The attibutes contain arrays which map element ids (row) and local degree of freedom of the element (column)
to global dof id (value).
If you want to get the global dof associated with element 5 and local element dof 0, run::

    >>> global_dof = my_assembly.element_indices[5,0]
    
The command

    >>> my_assembly.compute_element_indices()
    
generates or updates these indices and stores them in the property :py:attr:`element_indices<amfe.assembly.Assembly.element_indices>`.
This method updates the :py:attr:`neumann_indices<amfe.assembly.Assembly.neumann_indices>`, too.

Furthermore this method updates another property, called :py:attr:`elements_on_node<amfe.assembly.Assembly.elements_on_node>`.
This property is a vector which contains the number of elements that are associated with a certain node.
Example: If you want to know, how many elements are associated with node 143, run::

    >>> my_assembly.elements_on_node[143]


Assembling methods
------------------

For different tasks such as computation of eigenmodes, stress recovery or
nonlinear time integration you need different global entities.
The Assembly class provides assembly functions for many common combinations
of needed entities:

.. table:: Assembly methods

    +--------------------------------------------------------------------------------------------+------------------------------------------------------------------------------+
    | Method                                                                                     | Description                                                                  |
    +============================================================================================+==============================================================================+
    | :py:meth:`assemble_k_and_f(u,t)<amfe.assembly.Assembly.assemble_k_and_f>`                  | Return tangent stiffness and internal force vector                           |
    +--------------------------------------------------------------------------------------------+------------------------------------------------------------------------------+
    | :py:meth:`assemble_k_and_f_neumann(u,t)<amfe.assembly.Assembly.assemble_k_and_f_neumann>`  | Return tangent stiffness and force vector for all neumann boundary elements  |
    +--------------------------------------------------------------------------------------------+------------------------------------------------------------------------------+
    | :py:meth:`assemble_k_f_S_E(u,t)<amfe.assembly.Assembly.assemble_k_f_S_E>`                  | Return tangent stiffness, internal force vector and strains and stresses     |
    +--------------------------------------------------------------------------------------------+------------------------------------------------------------------------------+
    | :py:meth:`assemble_m(u,t)<amfe.assembly.Assembly.assemble_m>`                              | Return mass matrix                                                           |
    +--------------------------------------------------------------------------------------------+------------------------------------------------------------------------------+

.. Note::

    In AMfe the strains and stresses returned by the assembly functions are nodal entities.
    The nodal strains and stresses are mean values of the neighbouring element strains and stresses.

**Example:**

Return the tangent stiffness matrix and internal force vector for global displacement vector u_global::

    >>> K, f_int = my_assembly.assemble_k_and_f(u_global) # parameter time t can be dropped if time independent system


The CSR-Format and helper functions
-----------------------------------

CSR-storage
^^^^^^^^^^^

For the mass and stiffness matrices the scipy csr-format is used for efficient
storage of those matrices.
The format is very powerful if you have to do operations such as addition
or multiplacation of matrices.
Changes in number and posisitions of nonzero entries are expensive. This is why
it is highly recommended to preallocate a CSR matrix by using
:py:meth:`preallocate_csr()<amfe.assembly.Assembly.preallocate_csr>`-method.


The CSR-format consists of three vectors: data, indices and indptr.
The data vector contains the values of the nonzero elements.
The indices and indptr vectors map these data to the positions in the matrix pattern.

**Example:**

    You want to store the matrix
    
    | [10 0 0 34
    |  0 16 17 23
    |  53 39 85 86
    |  71 0 0 91]
    
    in CSR-format::
    
        >>> import numpy as np
        >>> import scipy as sp
        >>> data = np.array([10, 34, 16, 17, 23, 53, 39, 85, 86, 71, 91])
        >>> indptr = np.array([0, 2, 5, 9, 11])
        >>> indices = np.array([0, 3, 1, 2, 3, 0, 1, 2, 3, 0, 3])
        >>> A = sp.sparse.csr_matrix((data, indices, indptr))
        >>> # Check
        >>> B = A.toarray()
        >>> print(B)
        [[10  0  0 34]
         [ 0 16 17 23]
         [53 39 85 86]
         [71  0  0 91]]






Altering CSR matrices
^^^^^^^^^^^^^^^^^^^^^

Beside the preallocation method there are two helper functions in the assembly
module that can help using the CSR storage format.

The first helper function get_index_of_csr_data(i,j,indptr,indices)
returns the index where a certain matrix entry of the (i-1)-th row and (j-1)-th column
is stored in the data vector::

    >>> get_index_of_csr_data(i,j,indptr,indices)

.. important::

    The indexing starts at zero. Thus, if you want to access the 3rd row and 5th
    column for example, you have to pass i=2, j=4.
    

Example:

Get the value of the third row, second column of the matrix A from example above::

    >>> index = amfe.assembly.get_index_of_csr_data(2,1,indptr,indices)
    >>> print(index)
    6
    >>> value = data[index]
    >>> print(value)
    39


The second helper function is

    >>> fill_csr_matrix(indptr, indices, vals, K, k_indices)
    
This function helps to assemble local element matrices into global matrices.
The first parameters are the vectors of the csr-matrix which shall be altered.
The parameter K is the local element matrix and the vector k_indices is the
vector with the global indices (the global dofs) where the local matrix has to be assembled to.

.. note::
    
    Please note again that indexing starts at zero.



