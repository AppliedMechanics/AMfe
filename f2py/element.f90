

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
            B(:,(i-1)*3+1) = (/ F(1,1)*Bt(1,i), F(1,2)*Bt(2,i), F(1,3)*Bt(3,i), &
               F(1,2)*Bt(3,i) + F(1,3)*Bt(2,i), F(1,3)*Bt(1,i) + F(1,1)*Bt(3,i), F(1,1)*Bt(2,i) + F(1,2)*Bt(1,i) /)
            B(:,(i-1)*3+2) = (/ F(2,1)*Bt(1,i), F(2,2)*Bt(2,i), F(2,3)*Bt(3,i), &
               F(2,2)*Bt(3,i) + F(2,3)*Bt(2,i), F(2,3)*Bt(1,i) + F(2,1)*Bt(3,i), F(2,1)*Bt(2,i) + F(2,2)*Bt(1,i) /)
            B(:,(i-1)*3+3) = (/ F(3,1)*Bt(1,i), F(3,2)*Bt(2,i), F(3,3)*Bt(3,i), &
               F(3,2)*Bt(3,i) + F(3,3)*Bt(2,i), F(3,3)*Bt(1,i) + F(3,1)*Bt(3,i), F(3,1)*Bt(2,i) + F(3,2)*Bt(1,i) /)
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

    u_e = transpose(reshape(u, (/2, 3/)))
    
    A0 = 0.5*((X3-X2)*(Y1-Y2) - (X1-X2)*(Y3-Y2))

    EYE = reshape((/1, 0, 0, 1/), shape(EYE))

    B0_tilde(1,1) = (Y2-Y3) / (2*A0)
    B0_tilde(1,2) = (Y3-Y1) / (2*A0)
    B0_tilde(1,3) = (Y1-Y2) / (2*A0)
    B0_tilde(2,1) = (X3-X2) / (2*A0)
    B0_tilde(2,2) = (X1-X3) / (2*A0)
    B0_tilde(2,3) = (X2-X1) / (2*A0)

!    The standard procedure for doing Total Lagrangian
!   computing the 'help' matrix H
    H = matmul(transpose(u_e), transpose(B0_tilde))
!     The deformation gradient F
    F = H + EYE
!     The Green-Lagrange-Strain tensor
    E = 0.5*(H + transpose(H) + matmul(transpose(H), H))
!     The Green-Lagrange-Strain tensor in voigt notation
    E_v = (/ E(1,1), E(2,2), 2*E(1,2) /)
!     The second Piola-Kirchhoff Stress tensor in voigt notatoin
    S_v = matmul(C_SE, E_v)
!     The second Piola-Kirchhoff stress tensor in matrix notation
    S = reshape((/ S_v(1), S_v(3), S_v(3), S_v(2) /), shape(S))
!     The variational matrix B0
    call compute_b_matrix(B0_tilde, F, B0, 3, 2)
!     The material stiffness
    K_mat = matmul(transpose(B0), matmul(C_SE, B0))*A0*t
!     And the geometric stiffness
    K_geo_sm = matmul(transpose(B0_tilde), matmul(S, B0_tilde))*A0*t
    call scatter_matrix(K_geo_sm, K_geo, 2, 3, 3)
    K = K_mat + K_geo
!     and last but not least the internal force
    f_int = matmul(transpose(B0), S_v)*A0*t

end subroutine


subroutine tri6_k_and_f(X, u, C_SE, K, f_int, t)
    implicit none

    integer :: i
    real(8), intent(in) :: X(12), u(12), C_SE(3,3), t
    real(8), intent(out) :: K(12, 12), f_int(12)
    real(8) :: X1, X2, X3, Y1, Y2, Y3, X4, Y4, X5, Y5, X6, Y6, A0
    real(8) :: Jx1 ,Jx2 ,Jx3 ,Jy1 ,Jy2 ,Jy3, w
    real(8) :: u_e(6,2)
    real(8) :: K_geo_sm(6,6), K_mat(12,12), K_geo(12,12)
    real(8) :: B0_tilde(2,6), B0(3,12)
    real(8) :: E(2,2), H(2,2), F(2,2), EYE(2,2), S(2,2), S_v(3), E_v(3)

    real(8) :: gauss_points(4,3), weights(4)
    real(8) :: L1, L2, L3, det

!   External functions that will be used afterwards
    external :: scatter_matrix
    external :: compute_b_matrix


    X1 = X(1)
    Y1 = X(2)
    X2 = X(3)
    Y2 = X(4)
    X3 = X(5)
    Y3 = X(6)
    X4 = X(7)
    Y4 = X(8)
    X5 = X(9)
    Y5 = X(10)
    X6 = X(11)
    Y6 = X(12)

    ! take care of the fortran matrix order (columns first!): 
    u_e = transpose(reshape(u, (/2, 6/)))

    ! take care of a precise description of 1/3 in order to avoid errors!
    weights = (/ -27./48.0D0, 25./48.0D0, 25./48.0D0, 25./48.0D0 /)
    gauss_points(1,:) = (/ 1/3.0D0, 1/3.0D0, 1/3.0D0 /)
    gauss_points(2,:) = (/0.6D0, 0.2D0, 0.2D0 /)
    gauss_points(3,:) = (/0.2D0, 0.6D0, 0.2D0 /)
    gauss_points(4,:) = (/0.2D0, 0.2D0, 0.6D0 /)

    det = X1*Y2 - X1*Y3 - X2*Y1 + X2*Y3 + X3*Y1 - X3*Y2
    A0 = det/2
    EYE = reshape((/1, 0, 0, 1/), shape(EYE))

    K = 0.0
    f_int = 0.0

    ! loop over all quadrature points
    do i=1,4
       L1 = gauss_points(i, 1)
       L2 = gauss_points(i, 2)
       L3 = gauss_points(i, 3)
       w  = weights(i)

       ! the entries in the jacobian dX_dL
       Jx1 = 4*L2*X4 + 4*L3*X6 + X1*(4*L1 - 1)
       Jx2 = 4*L1*X4 + 4*L3*X5 + X2*(4*L2 - 1)
       Jx3 = 4*L1*X6 + 4*L2*X5 + X3*(4*L3 - 1)
       Jy1 = 4*L2*Y4 + 4*L3*Y6 + Y1*(4*L1 - 1)
       Jy2 = 4*L1*Y4 + 4*L3*Y5 + Y2*(4*L2 - 1)
       Jy3 = 4*L1*Y6 + 4*L2*Y5 + Y3*(4*L3 - 1)

       det = Jx1*Jy2 - Jx1*Jy3 - Jx2*Jy1 + Jx2*Jy3 + Jx3*Jy1 - Jx3*Jy2
       
       B0_tilde(1,:) = (/            (Jy2 - Jy3)*(4*L1 - 1), &
                                    (-Jy1 + Jy3)*(4*L2 - 1), &
                                     (Jy1 - Jy2)*(4*L3 - 1), &
                       4*L1*(-Jy1 + Jy3) + 4*L2*(Jy2 - Jy3), &
                       4*L2*(Jy1 - Jy2) + 4*L3*(-Jy1 + Jy3), &
                        4*L1*(Jy1 - Jy2) + 4*L3*(Jy2 - Jy3) /)
        
       B0_tilde(2,:) = (/           (-Jx2 + Jx3)*(4*L1 - 1), &
                                     (Jx1 - Jx3)*(4*L2 - 1), &
                                    (-Jx1 + Jx2)*(4*L3 - 1), &
                       4*L1*(Jx1 - Jx3) + 4*L2*(-Jx2 + Jx3), &
                       4*L2*(-Jx1 + Jx2) + 4*L3*(Jx1 - Jx3), &
                      4*L1*(-Jx1 + Jx2) + 4*L3*(-Jx2 + Jx3) /)

       H = matmul(transpose(u_e), transpose(B0_tilde))
       F = H + EYE
       E = 0.5*(H + transpose(H) + matmul(transpose(H), H))
       E_v = (/ E(1,1), E(2,2), 2*E(1,2) /)
       S_v = matmul(C_SE, E_v)
       S = reshape((/ S_v(1), S_v(3), S_v(3), S_v(2) /), shape(S))

       call compute_b_matrix(B0_tilde, F, B0, 6, 2)

       K_mat = matmul(transpose(B0), matmul(C_SE, B0))*A0*t
       K_geo_sm = matmul(transpose(B0_tilde), matmul(S, B0_tilde))*A0*t

       call scatter_matrix(K_geo_sm, K_geo, 2, 6, 6)

       K = K + weights(i)*(K_mat + K_geo)
       f_int = f_int + weights(i)*matmul(transpose(B0), S_v)*A0*t

    end do
end subroutine


