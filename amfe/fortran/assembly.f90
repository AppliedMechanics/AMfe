! Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
! Universitaet Muenchen.
!
! Distributed under BSD-3-Clause License. See LICENSE-File for more information
!


subroutine get_index_of_csr_data(i, j, indptr, indices, k, n, m)
    ! i, j are the indices of the global array
    ! indptr and indices are the arrays from the csr matrix
    ! k is the python-index of the data array in the csr matrix
    ! n is the dimension of the indptr-array
    ! m is the dimension of the indices array

    ! be careful, but the indexing is given in numpy arrays.
    ! So the entries containing indices start at zero!

    implicit none

    integer, intent(in) :: n, m
    integer, intent(in) :: i, j, indptr(n), indices(m)
    integer, intent(out) :: k

    k = indptr(i+1)
    do while( j /= indices(k+1) )
        k = k + 1

        if (k > indptr(i+2)) then
            k = 0
            exit
        end if

    end do

end subroutine




subroutine fill_csr_matrix(indptr, indices, vals, Mat, k_indices, N, M, o)
    implicit none

    ! Mat: local element (stiffness) matrix
    ! k_indices: array of the indices mapping the local dofs to the global dofs

    ! N: Number of rows and columns in the CSR-Matrix
    ! M: Number of nonzero entries in the CSR-Matrix
    ! o: Number of rows or columns of the 'small' K-matrix, is also the number of the k_indices

    integer, intent(in) :: N, M, o
    real(8), intent(inout) :: vals(M)
    real(8), intent(in) :: Mat(o, o)
    integer, intent(in) :: indptr(N), indices(M), k_indices(o)
    integer :: i, j, k

    external :: get_index_of_csr_data

    ! loop over the indices of K
    do i=0,o-1
        do j=0,o-1
            call get_index_of_csr_data(k_indices(i+1), k_indices(j+1), indptr, indices, k, N, M)
            vals(k+1) = vals(k+1) + Mat(i+1, j+1)
        end do
    end do

end subroutine
