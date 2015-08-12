

subroutine scatter_matrix(A, B, ndim, n, m)
    implicit none
    integer, intent(in) :: n, m, ndim
    real(8), intent(in) :: A(m, n)
    real(8), intent(out) :: B(ndim*m, ndim*n)
    integer i, j, k

!    Initialization to zero; otherwise random stuff is there...
    B(:,:) = 0.0
! looping style like in Python:
    do i=0,m-1
        do j=0,n-1
            do k=0,ndim-1
                B(ndim*i+k+1,ndim*j+k+1) = A(i+1,j+1)
            end do
        end do
    end do

end subroutine


subroutine compute_B_matrix(Bt, F, B, no_of_nodes, no_of_dims)
    implicit none

    integer, intent(in)  :: no_of_nodes, no_of_dims
    real(8), intent(in)  :: F(no_of_dims,no_of_dims), Bt(no_of_dims, no_of_nodes)
    real(8), intent(out) :: B(no_of_dims*(no_of_dims + 1)/2, no_of_nodes*no_of_dims)
    integer :: i


    if (no_of_dims.eq.2) then
!        here the counting starts at 1 in order to make indexing for Bt easier
        do i=1,no_of_nodes
            B(:,(i-1)*2+1)   = (/ F(1,1)*Bt(1,i), F(1,2)*Bt(2,i), F(1,1)*Bt(2,i) + F(1,2)*Bt(1,i) /)
            B(:,(i-1)*2+2) = (/ F(2,1)*Bt(1,i), F(2,2)*Bt(2,i), F(2,1)*Bt(2,i) + F(2,2)*Bt(1,i) /)
        end do
    elseif (no_of_dims.eq.3) then
        do i=1,no_of_nodes
            B(:,(i-1)*2+1) = (/ F(1,1)*Bt(1,i), F(1,2)*Bt(2,i), F(1,3)*Bt(3,i), &
               F(1,1)*Bt(2,i) + F(1,2)*Bt(1,i), F(1,2)*Bt(3,i) + F(1,3)*Bt(2,i), F(1,3)*Bt(1,i) + F(1,1)*Bt(3,i) /)
            B(:,(i-1)*2+2) = (/ F(2,1)*Bt(1,i), F(2,2)*Bt(2,i), F(2,3)*Bt(3,i), &
               F(2,1)*Bt(2,i) + F(2,2)*Bt(1,i), F(2,2)*Bt(3,i) + F(2,3)*Bt(2,i), F(2,3)*Bt(1,i) + F(2,1)*Bt(3,i) /)
            B(:,(i-1)*2+3) = (/ F(3,1)*Bt(1,i), F(3,2)*Bt(2,i), F(3,3)*Bt(3,i), &
               F(3,1)*Bt(2,i) + F(3,2)*Bt(1,i), F(3,2)*Bt(3,i) + F(3,3)*Bt(2,i), F(3,3)*Bt(1,i) + F(3,1)*Bt(3,i) /)
        end do
    else
        B = B*0
    endif

end subroutine



subroutine tri3_k_and_f(X, u, C_SE, K, f_int, t)
    implicit none

!   Input and output parameters

!   What is important here is that the
    real(8), intent(in) :: X(6), u(6), C_SE(3,3), t
    real(8), intent(out) :: K(6, 6), f_int(6)
    real(8) :: X1, X2, X3, Y1, Y2, Y3, A0
    real(8) :: u_e(3,2)
    real(8) :: K_geo_sm(3,3), K_mat(6,6), K_geo(6,6)
    real(8) :: B0_tilde(2,3),  B0(3,6)
    real(8) :: E(2,2), H(2,2), F(2,2), EYE(2,2), S(2,2), S_v(3), E_v(3)
!   External functions that will be used afterwards
    external :: scatter_matrix
    external :: compute_b_matrix

!   The variables for calculation

    X1 = X(1)
    Y1 = X(2)
    X2 = X(3)
    Y2 = X(4)
    X3 = X(5)
    Y3 = X(6)

    u_e = reshape(u, (/3,2/))

    A0 = 0.5*((X3-X2)*(Y1-Y2) - (X1-X2)*(Y3-Y2))

    EYE = reshape((/1, 0, 0, 1/), shape(EYE))

    B0_tilde(1,1) = (Y2-Y3) / (2*A0)
    B0_tilde(1,2) = (Y3-Y1) / (2*A0)
    B0_tilde(1,3) = (Y1-Y2) / (2*A0)
    B0_tilde(2,1) = (X3-X2) / (2*A0)
    B0_tilde(2,2) = (X1-X3) / (2*A0)
    B0_tilde(2,3) = (X2-X1) / (2*A0)

!    The standard procedure for doing Total Lagrangian
    H = matmul(transpose(u_e), transpose(B0_tilde))
    F = H + EYE
    E = 0.5*(H + transpose(H) + matmul(transpose(H), H))
    E_v = (/ E(1,1), E(2,2), 2*E(1,2) /)
    S_v = matmul(C_SE, E_v)
    S = reshape((/ S_v(1), S_v(3), S_v(3), S_v(2) /), shape(S))
    call compute_b_matrix(B0_tilde, F, B0, 3, 2)
    K_mat = matmul(transpose(B0), matmul(C_SE, B0))*A0*t
    K_geo_sm = matmul(transpose(B0_tilde), matmul(S, B0_tilde))*A0*t
    call scatter_matrix(K_geo_sm, K_geo, 2, 3, 3)
    K = K_mat + K_geo
    f_int = matmul(transpose(B0), S_v)*A0*t

end subroutine