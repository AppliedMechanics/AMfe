! Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
! Universitaet Muenchen.
!
! Distributed under BSD-3-Clause License. See LICENSE-File for more information
!


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
    real(8), intent(in)  :: F(no_of_dims,no_of_dims), Bt(no_of_nodes, no_of_dims)
    real(8), intent(out) :: B(no_of_dims*(no_of_dims + 1)/2, no_of_nodes*no_of_dims)
    integer :: i


    if (no_of_dims.eq.2) then
!        here the counting starts at 1 in order to make indexing for Bt easier
        do i=1,no_of_nodes
            B(:,(i-1)*2+1)   = (/ F(1,1)*Bt(i,1), F(1,2)*Bt(i,2), F(1,1)*Bt(i,2) + F(1,2)*Bt(i,1) /)
            B(:,(i-1)*2+2) = (/ F(2,1)*Bt(i,1), F(2,2)*Bt(i,2), F(2,1)*Bt(i,2) + F(2,2)*Bt(i,1) /)
        end do
    elseif (no_of_dims.eq.3) then
        do i=1,no_of_nodes
            B(:,(i-1)*3+1) = (/ F(1,1)*Bt(i,1), F(1,2)*Bt(i,2), F(1,3)*Bt(i,3), &
               F(1,2)*Bt(i,3) + F(1,3)*Bt(i,2), F(1,3)*Bt(i,1) + F(1,1)*Bt(i,3), F(1,1)*Bt(i,2) + F(1,2)*Bt(i,1) /)
            B(:,(i-1)*3+2) = (/ F(2,1)*Bt(i,1), F(2,2)*Bt(i,2), F(2,3)*Bt(i,3), &
               F(2,2)*Bt(i,3) + F(2,3)*Bt(i,2), F(2,3)*Bt(i,1) + F(2,1)*Bt(i,3), F(2,1)*Bt(i,2) + F(2,2)*Bt(i,1) /)
            B(:,(i-1)*3+3) = (/ F(3,1)*Bt(i,1), F(3,2)*Bt(i,2), F(3,3)*Bt(i,3), &
               F(3,2)*Bt(i,3) + F(3,3)*Bt(i,2), F(3,3)*Bt(i,1) + F(3,1)*Bt(i,3), F(3,1)*Bt(i,2) + F(3,2)*Bt(i,1) /)
        end do
    else
        B = B*0
    endif

end subroutine

subroutine invert_3_by_3_matrix(A, A_inv, det)
    ! Invert the matrix A and compute the determinant
    implicit none

    real(8), intent(in) :: A(3,3)
    real(8), intent(out) :: A_inv(3,3), det
    real(8) :: a11, a12, a13, a21, a22, a23, a31, a32, a33
    a11 = A(1,1)
    a12 = A(1,2)
    a13 = A(1,3)
    a21 = A(2,1)
    a22 = A(2,2)
    a23 = A(2,3)
    a31 = A(3,1)
    a32 = A(3,2)
    a33 = A(3,3)
    det = a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31
    A_inv = reshape((/ a22*a33 - a23*a32, -a21*a33 + a23*a31, &
                       a21*a32 - a22*a31, -a12*a33 + a13*a32, &
                       a11*a33 - a13*a31, -a11*a32 + a12*a31, &
                       a12*a23 - a13*a22, -a11*a23 + a13*a21, &
                       a11*a22 - a12*a21 /), (/3,3/)) / det
end subroutine

subroutine tri3_k_f_s_e(X, u, K, f_int, t, S_exp, E_exp, S_Sv_and_C_2d)
    implicit none

!   What is important here is that the
    real(8), intent(in) :: X(6), u(6), t
    real(8), intent(out) :: K(6, 6), f_int(6), S_exp(3, 6), E_exp(3, 6)
    real(8) :: X1, X2, X3, Y1, Y2, Y3, A0
    real(8) :: u_e(3,2), C_SE(3,3)
    real(8) :: K_geo_sm(3,3), K_mat(6,6), K_geo(6,6)
    real(8) :: B0_tilde(3,2),  B0(3,6)
    real(8) :: E(2,2), H(2,2), F(2,2), EYE(2,2), S(2,2), S_v(3)
    real(8) :: extrapol(3,1)
!   External functions that will be used afterwards
    external :: scatter_matrix
    external :: compute_b_matrix
    external :: S_Sv_and_C

!   The variables for calculation

    X1 = X(1)
    Y1 = X(2)
    X2 = X(3)
    Y2 = X(4)
    X3 = X(5)
    Y3 = X(6)

    u_e = transpose(reshape(u, (/2, 3/)))
    extrapol = reshape( (/ 1.0D0, 1.0D0, 1.0D0 /) , (/3, 1/))

    A0 = 0.5*((X3-X2)*(Y1-Y2) - (X1-X2)*(Y3-Y2))

    EYE = reshape((/1, 0, 0, 1/), shape(EYE))

    B0_tilde(1,1) = (Y2-Y3) / (2*A0)
    B0_tilde(2,1) = (Y3-Y1) / (2*A0)
    B0_tilde(3,1) = (Y1-Y2) / (2*A0)
    B0_tilde(1,2) = (X3-X2) / (2*A0)
    B0_tilde(2,2) = (X1-X3) / (2*A0)
    B0_tilde(3,2) = (X2-X1) / (2*A0)

!    The standard procedure for doing Total Lagrangian
!   computing the 'help' matrix H
    H = matmul(transpose(u_e), B0_tilde)
!     The deformation gradient F
    F = H + EYE
!     The Green-Lagrange-Strain tensor
    E = 0.5*(H + transpose(H) + matmul(transpose(H), H))
!     Compute the stress and the tangent moduli via external call
    call S_Sv_and_C_2d(E, S, S_v, C_SE)
!     The variational matrix B0
    call compute_b_matrix(B0_tilde, F, B0, 3, 2)
!     The material stiffness
    K_mat = matmul(transpose(B0), matmul(C_SE, B0))*A0*t
!     And the geometric stiffness
    K_geo_sm = matmul(B0_tilde, matmul(S, transpose(B0_tilde)))*A0*t
    call scatter_matrix(K_geo_sm, K_geo, 2, 3, 3)
    K = K_mat + K_geo
!     and last but not least the internal force
    f_int = matmul(transpose(B0), S_v)*A0*t
    S_exp = matmul(extrapol, reshape((/S(1,1), S(1,2), 0.0D0, S(2,2), 0.0D0, 0.0D0/), (/1, 6/)))
    E_exp = matmul(extrapol, reshape((/ E(1,1), E(1,2), 0.0D0, E(2,2), 0.0D0, 0.0D0/), (/1, 6/)))

end subroutine tri3_k_f_s_e


subroutine tri6_k_f_s_e(X, u, K, f_int, t, S_exp, E_exp, S_Sv_and_C_2d)
    implicit none

    integer :: i
    real(8), intent(in) :: X(12), u(12), t
    real(8), intent(out) :: K(12, 12), f_int(12), S_exp(6,6), E_exp(6,6)
    real(8) :: X1, X2, X3, Y1, Y2, Y3, X4, Y4, X5, Y5, X6, Y6, A0
    real(8) :: Jx1 ,Jx2 ,Jx3 ,Jy1 ,Jy2 ,Jy3, w
    real(8) :: u_e(6,2), C_SE(3,3)
    real(8) :: K_geo_sm(6,6), K_mat(12,12), K_geo(12,12)
    real(8) :: B0_tilde(6,2), B0(3,12)
    real(8) :: E(2,2), H(2,2), F(2,2), EYE(2,2), S(2,2), S_v(3)

    real(8) :: gauss_points(3,3), weights(3), extrapol(6,3)
!    real(8) :: w0, w1, w2, alpha1, alpha2, beta1, beta2
    real(8) :: L1, L2, L3, det

!   External functions that will be used afterwards
    external :: scatter_matrix
    external :: compute_b_matrix
    external :: S_Sv_and_C_2d

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
    extrapol(:,1) = (/ 5/3.0D0, -1/3.0D0, -1/3.0D0, 2/3.0D0, -1/3.0D0, 2/3.0D0 /)
    extrapol(:,2) = (/-1/3.0D0, 5/3.0D0, -1/3.0D0, 2/3.0D0, 2/3.0D0, -1/3.0D0 /)
    extrapol(:,3) = (/ -1/3.0D0, -1/3.0D0, 5/3.0D0, -1/3.0D0, 2/3.0D0, 2/3.0D0 /)

!    ! take care of a precise description of 1/3 in order to avoid errors!
!    w0 = 0.225D0
!
!    alpha1 = 0.0597158717D0
!    beta1 = 0.4701420641D0
!    w1 = 0.1323941527D0
!
!    alpha2 = 0.7974269853D0
!    beta2 = 0.1012865073D0
!    w2 = 0.1259391805D0
!
!    weights = (/ w0, w1, w1, w1, w2, w2, w2 /)
!    gauss_points(1,:) = (/ 1/3.0D0, 1/3.0D0, 1/3.0D0 /)
!    gauss_points(2,:) = (/ alpha1, beta1, beta1 /)
!    gauss_points(3,:) = (/ beta1, alpha1, beta1 /)
!    gauss_points(4,:) = (/ beta1, beta1, alpha1 /)
!    gauss_points(5,:) = (/ alpha2, beta2, beta2 /)
!    gauss_points(6,:) = (/ beta2, alpha2, beta2 /)
!    gauss_points(7,:) = (/ beta2, beta2, alpha2 /)

    weights = (/ 1/3.0D0, 1/3.0D0, 1/3.0D0 /)
    gauss_points(1,:) = (/ 2/3.0D0, 1/6.0D0, 1/6.0D0 /)
    gauss_points(2,:) = (/ 1/6.0D0, 2/3.0D0, 1/6.0D0 /)
    gauss_points(3,:) = (/ 1/6.0D0, 1/6.0D0, 2/3.0D0 /)

    EYE = reshape((/1, 0, 0, 1/), shape(EYE))

    K = 0.0
    f_int = 0.0
    S_exp = 0.0
    E_exp = 0.0

    ! loop over all quadrature points
    do i=1,3
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
       A0 = det/2

       B0_tilde(:,1) = (/            (Jy2 - Jy3)*(4*L1 - 1), &
                                    (-Jy1 + Jy3)*(4*L2 - 1), &
                                     (Jy1 - Jy2)*(4*L3 - 1), &
                       4*L1*(-Jy1 + Jy3) + 4*L2*(Jy2 - Jy3), &
                       4*L2*(Jy1 - Jy2) + 4*L3*(-Jy1 + Jy3), &
                        4*L1*(Jy1 - Jy2) + 4*L3*(Jy2 - Jy3) /)

       B0_tilde(:,2) = (/           (-Jx2 + Jx3)*(4*L1 - 1), &
                                     (Jx1 - Jx3)*(4*L2 - 1), &
                                    (-Jx1 + Jx2)*(4*L3 - 1), &
                       4*L1*(Jx1 - Jx3) + 4*L2*(-Jx2 + Jx3), &
                       4*L2*(-Jx1 + Jx2) + 4*L3*(Jx1 - Jx3), &
                      4*L1*(-Jx1 + Jx2) + 4*L3*(-Jx2 + Jx3) /)

       B0_tilde = B0_tilde/det

       H = matmul(transpose(u_e), B0_tilde)
       F = H + EYE
       E = 0.5*(H + transpose(H) + matmul(transpose(H), H))

       call S_Sv_and_C_2d(E, S, S_v, C_SE)

       call compute_b_matrix(B0_tilde, F, B0, 6, 2)

       K_mat = matmul(transpose(B0), matmul(C_SE, B0)) * A0 * t
       K_geo_sm = matmul(B0_tilde, matmul(S, transpose(B0_tilde))) * A0 * t

       call scatter_matrix(K_geo_sm, K_geo, 2, 6, 6)

       K = K + w*(K_mat + K_geo)
       f_int = f_int + w*matmul(transpose(B0), S_v)*A0*t

       S_exp = S_exp + matmul(reshape(extrapol(:,i), (/6,1/)), &
                reshape((/S(1,1), S(1,2), 0.0D0, S(2,2), 0.0D0, 0.0D0/), (/1, 6/)))
       E_exp = E_exp + matmul(reshape(extrapol(:,i), (/6,1/)), &
                reshape((/ E(1,1), E(1,2), 0.0D0, E(2,2), 0.0D0, 0.0D0/), (/1, 6/)))

    end do
end subroutine tri6_k_f_s_e


subroutine tri6_m(X, rho, t, M)
    implicit none

    integer :: i
    real(8), intent(in) :: X(12), t, rho
    real(8), intent(out) :: M(12,12)
    real(8) :: X1, X2, X3, Y1, Y2, Y3, X4, Y4, X5, Y5, X6, Y6
    real(8) :: Jx1 ,Jx2 ,Jx3 ,Jy1 ,Jy2 ,Jy3, w
    real(8) :: N(6,1), M_small(6,6)

    real(8) :: gauss_points(3,3), weights(3)
!    real(8) :: w0, w1, w2, alpha1, alpha2, beta1, beta2
    real(8) :: L1, L2, L3, det

!   External functions that will be used afterwards
    external :: scatter_matrix

!    ! take care of a precise description of 1/3 in order to avoid errors!
!    w0 = 0.225D0
!
!    alpha1 = 0.0597158717D0
!    beta1 = 0.4701420641D0
!    w1 = 0.1323941527D0
!
!    alpha2 = 0.7974269853D0
!    beta2 = 0.1012865073D0
!    w2 = 0.1259391805D0
!
!    weights = (/ w0, w1, w1, w1, w2, w2, w2 /)
!    gauss_points(1,:) = (/ 1/3.0D0, 1/3.0D0, 1/3.0D0 /)
!    gauss_points(2,:) = (/ alpha1, beta1, beta1 /)
!    gauss_points(3,:) = (/ beta1, alpha1, beta1 /)
!    gauss_points(4,:) = (/ beta1, beta1, alpha1 /)
!    gauss_points(5,:) = (/ alpha2, beta2, beta2 /)
!    gauss_points(6,:) = (/ beta2, alpha2, beta2 /)
!    gauss_points(7,:) = (/ beta2, beta2, alpha2 /)

    weights = (/ 1/3.0D0, 1/3.0D0, 1/3.0D0 /)
    gauss_points(1,:) = (/ 1/6.0D0, 1/6.0D0, 2/3.0D0 /)
    gauss_points(2,:) = (/ 1/6.0D0, 2/3.0D0, 1/6.0D0 /)
    gauss_points(3,:) = (/ 2/3.0D0, 1/6.0D0, 1/6.0D0 /)



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

    M_small = 0.0
    M = 0.0

    do i=1,3
       L1 = gauss_points(i, 1)
       L2 = gauss_points(i, 2)
       L3 = gauss_points(i, 3)
       w  = weights(i)

       Jx1 = 4*L2*X4 + 4*L3*X6 + X1*(4*L1 - 1)
       Jx2 = 4*L1*X4 + 4*L3*X5 + X2*(4*L2 - 1)
       Jx3 = 4*L1*X6 + 4*L2*X5 + X3*(4*L3 - 1)
       Jy1 = 4*L2*Y4 + 4*L3*Y6 + Y1*(4*L1 - 1)
       Jy2 = 4*L1*Y4 + 4*L3*Y5 + Y2*(4*L2 - 1)
       Jy3 = 4*L1*Y6 + 4*L2*Y5 + Y3*(4*L3 - 1)

       det = Jx1*Jy2 - Jx1*Jy3 - Jx2*Jy1 + Jx2*Jy3 + Jx3*Jy1 - Jx3*Jy2

       N(:,1) = (/ L1*(2*L1 - 1), L2*(2*L2 - 1), L3*(2*L3 - 1), 4*L1*L2, 4*L2*L3, 4*L1*L3 /)
       M_small = M_small + matmul(N, transpose(N)) * det/2 * rho * t * w
    end do

    call scatter_matrix(M_small, M, 2, 6, 6)

end subroutine


subroutine tet4_k_f_s_e(X, u, K, f_int, S_exp, E_exp, S_Sv_and_C)
    implicit none

    real(8), intent(in) :: X(12), u(12)
    real(8), intent(out) :: K(12, 12), f_int(12), S_exp(4,6), E_exp(4,6)
    real(8) :: X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4
    real(8) :: u_e(4,3), C_SE(6,6)
    real(8) :: K_geo_sm(4,4), K_mat(12,12), K_geo(12,12)
    real(8) :: B0_tilde(4,3), B0(6,12)
    real(8) :: E(3,3), H(3,3), F(3,3), EYE(3,3), S(3,3), S_v(6)
    real(8) :: det, extrapol(4,1)

!   External functions that will be used afterwards
    external :: scatter_matrix
    external :: compute_b_matrix
    external :: S_Sv_and_C

    X1 = X(1)
    Y1 = X(2)
    Z1 = X(3)
    X2 = X(4)
    Y2 = X(5)
    Z2 = X(6)
    X3 = X(7)
    Y3 = X(8)
    Z3 = X(9)
    X4 = X(10)
    Y4 = X(11)
    Z4 = X(12)

    EYE = 0.0D0
    EYE(1,1) = 1
    EYE(2,2) = 1
    EYE(3,3) = 1
    extrapol = 1.0
    S_exp = 0.0
    E_exp = 0.0

    u_e = transpose(reshape(u, (/ 3, 4 /)))
    det = -X1*Y2*Z3 + X1*Y2*Z4 + X1*Y3*Z2 - X1*Y3*Z4 - X1*Y4*Z2 + X1*Y4*Z3 &
         + X2*Y1*Z3 - X2*Y1*Z4 - X2*Y3*Z1 + X2*Y3*Z4 + X2*Y4*Z1 - X2*Y4*Z3 &
         - X3*Y1*Z2 + X3*Y1*Z4 + X3*Y2*Z1 - X3*Y2*Z4 - X3*Y4*Z1 + X3*Y4*Z2 &
         + X4*Y1*Z2 - X4*Y1*Z3 - X4*Y2*Z1 + X4*Y2*Z3 + X4*Y3*Z1 - X4*Y3*Z2

    B0_tilde(1,:) = (/ -Y2*Z3 + Y2*Z4 + Y3*Z2 - Y3*Z4 - Y4*Z2 + Y4*Z3, &
                        X2*Z3 - X2*Z4 - X3*Z2 + X3*Z4 + X4*Z2 - X4*Z3, &
                       -X2*Y3 + X2*Y4 + X3*Y2 - X3*Y4 - X4*Y2 + X4*Y3 /)
    B0_tilde(2,:) = (/  Y1*Z3 - Y1*Z4 - Y3*Z1 + Y3*Z4 + Y4*Z1 - Y4*Z3, &
                       -X1*Z3 + X1*Z4 + X3*Z1 - X3*Z4 - X4*Z1 + X4*Z3, &
                        X1*Y3 - X1*Y4 - X3*Y1 + X3*Y4 + X4*Y1 - X4*Y3 /)
    B0_tilde(3,:) = (/ -Y1*Z2 + Y1*Z4 + Y2*Z1 - Y2*Z4 - Y4*Z1 + Y4*Z2, &
                        X1*Z2 - X1*Z4 - X2*Z1 + X2*Z4 + X4*Z1 - X4*Z2, &
                       -X1*Y2 + X1*Y4 + X2*Y1 - X2*Y4 - X4*Y1 + X4*Y2 /)
    B0_tilde(4,:) = (/ Y1*Z2 - Y1*Z3 - Y2*Z1 + Y2*Z3 + Y3*Z1 - Y3*Z2, &
                      -X1*Z2 + X1*Z3 + X2*Z1 - X2*Z3 - X3*Z1 + X3*Z2, &
                       X1*Y2 - X1*Y3 - X2*Y1 + X2*Y3 + X3*Y1 - X3*Y2 /)

    B0_tilde = B0_tilde/det
    H = matmul(transpose(u_e), B0_tilde)
    F = H + EYE
    E = 0.5*(H + transpose(H) + matmul(transpose(H), H))

    call S_Sv_and_C(E, S, S_v, C_SE)
    call compute_b_matrix(B0_tilde, F, B0, 4, 3)

    K_mat = matmul(transpose(B0), matmul(C_SE, B0)) * det/6
    K_geo_sm = matmul(B0_tilde, matmul(S, transpose(B0_tilde))) * det/6

    call scatter_matrix(K_geo_sm, K_geo, 3, 4, 4)

    K = K_mat + K_geo
    f_int = matmul(transpose(B0), S_v) * det/6

    S_exp = matmul(extrapol, reshape((/S(1,1), S(1,2), S(1,3), S(2,2), S(2,3), S(3,3)/), (/1, 6/)))
    E_exp = matmul(extrapol, reshape((/ E(1,1), E(1,2), E(1,3), E(2,2), E(2,3), E(3,3)/), (/1, 6/)))

end subroutine tet4_k_f_s_e


subroutine tet10_k_f_s_e(X, u, K, f_int, S_exp, E_exp, S_Sv_and_C)
    implicit none

    real(8), intent(in) :: X(30), u(30)
    real(8), intent(out) :: K(30, 30), f_int(30), S_exp(10,6), E_exp(10,6)
    real(8) :: X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4, X5, Y5, Z5
    real(8) :: X6, Y6, Z6, X7, Y7, Z7, X8, Y8, Z8, X9, Y9, Z9, X10, Y10, Z10
    real(8) :: Jx1, Jx2, Jx3, Jx4, Jy1, Jy2, Jy3, Jy4, Jz1, Jz2, Jz3, Jz4
    real(8) :: a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4
    real(8) :: L1, L2, L3, L4
    real(8) :: u_e(10,3), C_SE(6,6)
    real(8) :: K_geo_sm(10,10), K_mat(30,30), K_geo(30,30)
    real(8) :: B0_tilde(10,3), B0(6,30)
    real(8) :: E(3,3), H(3,3), F(3,3), EYE(3,3), S(3,3), S_v(6)
    real(8) :: det, extrapol(10,4)
    real(8) :: gauss_points(4,4), weights(4), a, b, w, m1, m2
    integer :: i

!   External functions that will be used afterwards
    external :: scatter_matrix
    external :: compute_b_matrix
    external :: S_Sv_and_C

    X1 = X(1)
    Y1 = X(2)
    Z1 = X(3)
    X2 = X(4)
    Y2 = X(5)
    Z2 = X(6)
    X3 = X(7)
    Y3 = X(8)
    Z3 = X(9)
    X4 = X(10)
    Y4 = X(11)
    Z4 = X(12)
    X5 = X(13)
    Y5 = X(14)
    Z5 = X(15)
    X6 = X(16)
    Y6 = X(17)
    Z6 = X(18)
    X7 = X(19)
    Y7 = X(20)
    Z7 = X(21)
    X8 = X(22)
    Y8 = X(23)
    Z8 = X(24)
    X9 = X(25)
    Y9 = X(26)
    Z9 = X(27)
    X10 = X(28)
    Y10 = X(29)
    Z10 = X(30)

    EYE = 0.0D0
    EYE(1,1) = 1
    EYE(2,2) = 1
    EYE(3,3) = 1

    ! take care of the fortran matrix order (columns first!):
    u_e = transpose(reshape(u, (/3, 10/)))

    ! take care of a precise description of 1/3 in order to avoid errors!
    a = 0.13819660112501050D0
    b = 0.585410196624968520D0
    w = 1/4D0

    gauss_points(1,:) = (/ b, a, a, a /)
    gauss_points(2,:) = (/ a, b, a, a /)
    gauss_points(3,:) = (/ a, a, b, a /)
    gauss_points(4,:) = (/ a, a, a, b /)

    weights = (/ w, w, w, w /)

    c1 = 1/4.0D0 + 3.0D0*sqrt(5.0D0)/4.0D0 ! close corner node
    c2 = -sqrt(5.0D0)/4.0D0 + 1/4.0D0  ! far corner node
    m1 = 1/4.0D0 + sqrt(5.0D0)/4.0D0   ! close mid-node
    m2 = -sqrt(5.0D0)/4.0D0 + 1/4.0D0  ! far mid node


    extrapol(:,1) = (/ c1, c2, c2, c2, m1, m2, m1, m1, m2, m2 /)
    extrapol(:,2) = (/ c2, c1, c2, c2, m1, m1, m2, m2, m1, m2 /)
    extrapol(:,3) = (/ c2, c2, c1, c2, m2, m1, m1, m2, m2, m1 /)
    extrapol(:,4) = (/ c2, c2, c2, c1, m2, m2, m2, m1, m1, m1 /)



    ! set the matrices and vectors to zero
    K = 0.0
    f_int = 0.0
    S_exp = 0.0
    E_exp = 0.0

    ! Loop over the gauss points
    do i = 1, 4

        L1 = gauss_points(i, 1)
        L2 = gauss_points(i, 2)
        L3 = gauss_points(i, 3)
        L4 = gauss_points(i, 4)
        w  = weights(i)

        Jx1 = 4*L2*X5 + 4*L3*X7 + 4*L4*X8  + X1*(4*L1 - 1)
        Jx2 = 4*L1*X5 + 4*L3*X6 + 4*L4*X9  + X2*(4*L2 - 1)
        Jx3 = 4*L1*X7 + 4*L2*X6 + 4*L4*X10 + X3*(4*L3 - 1)
        Jx4 = 4*L1*X8 + 4*L2*X9 + 4*L3*X10 + X4*(4*L4 - 1)
        Jy1 = 4*L2*Y5 + 4*L3*Y7 + 4*L4*Y8  + Y1*(4*L1 - 1)
        Jy2 = 4*L1*Y5 + 4*L3*Y6 + 4*L4*Y9  + Y2*(4*L2 - 1)
        Jy3 = 4*L1*Y7 + 4*L2*Y6 + 4*L4*Y10 + Y3*(4*L3 - 1)
        Jy4 = 4*L1*Y8 + 4*L2*Y9 + 4*L3*Y10 + Y4*(4*L4 - 1)
        Jz1 = 4*L2*Z5 + 4*L3*Z7 + 4*L4*Z8  + Z1*(4*L1 - 1)
        Jz2 = 4*L1*Z5 + 4*L3*Z6 + 4*L4*Z9  + Z2*(4*L2 - 1)
        Jz3 = 4*L1*Z7 + 4*L2*Z6 + 4*L4*Z10 + Z3*(4*L3 - 1)
        Jz4 = 4*L1*Z8 + 4*L2*Z9 + 4*L3*Z10 + Z4*(4*L4 - 1)


        det = - Jx1*Jy2*Jz3 + Jx1*Jy2*Jz4 + Jx1*Jy3*Jz2 - Jx1*Jy3*Jz4 &
              - Jx1*Jy4*Jz2 + Jx1*Jy4*Jz3 + Jx2*Jy1*Jz3 - Jx2*Jy1*Jz4 &
              - Jx2*Jy3*Jz1 + Jx2*Jy3*Jz4 + Jx2*Jy4*Jz1 - Jx2*Jy4*Jz3 &
              - Jx3*Jy1*Jz2 + Jx3*Jy1*Jz4 + Jx3*Jy2*Jz1 - Jx3*Jy2*Jz4 &
              - Jx3*Jy4*Jz1 + Jx3*Jy4*Jz2 + Jx4*Jy1*Jz2 - Jx4*Jy1*Jz3 &
              - Jx4*Jy2*Jz1 + Jx4*Jy2*Jz3 + Jx4*Jy3*Jz1 - Jx4*Jy3*Jz2

        a1 = -Jy2*Jz3 + Jy2*Jz4 + Jy3*Jz2 - Jy3*Jz4 - Jy4*Jz2 + Jy4*Jz3
        a2 =  Jy1*Jz3 - Jy1*Jz4 - Jy3*Jz1 + Jy3*Jz4 + Jy4*Jz1 - Jy4*Jz3
        a3 = -Jy1*Jz2 + Jy1*Jz4 + Jy2*Jz1 - Jy2*Jz4 - Jy4*Jz1 + Jy4*Jz2
        a4 =  Jy1*Jz2 - Jy1*Jz3 - Jy2*Jz1 + Jy2*Jz3 + Jy3*Jz1 - Jy3*Jz2
        b1 =  Jx2*Jz3 - Jx2*Jz4 - Jx3*Jz2 + Jx3*Jz4 + Jx4*Jz2 - Jx4*Jz3
        b2 = -Jx1*Jz3 + Jx1*Jz4 + Jx3*Jz1 - Jx3*Jz4 - Jx4*Jz1 + Jx4*Jz3
        b3 =  Jx1*Jz2 - Jx1*Jz4 - Jx2*Jz1 + Jx2*Jz4 + Jx4*Jz1 - Jx4*Jz2
        b4 = -Jx1*Jz2 + Jx1*Jz3 + Jx2*Jz1 - Jx2*Jz3 - Jx3*Jz1 + Jx3*Jz2
        c1 = -Jx2*Jy3 + Jx2*Jy4 + Jx3*Jy2 - Jx3*Jy4 - Jx4*Jy2 + Jx4*Jy3
        c2 =  Jx1*Jy3 - Jx1*Jy4 - Jx3*Jy1 + Jx3*Jy4 + Jx4*Jy1 - Jx4*Jy3
        c3 = -Jx1*Jy2 + Jx1*Jy4 + Jx2*Jy1 - Jx2*Jy4 - Jx4*Jy1 + Jx4*Jy2
        c4 =  Jx1*Jy2 - Jx1*Jy3 - Jx2*Jy1 + Jx2*Jy3 + Jx3*Jy1 - Jx3*Jy2

        B0_tilde(1,:) = (/     a1*(4*L1 - 1),     b1*(4*L1 - 1), c1*(4*L1 - 1) /)
        B0_tilde(2,:) = (/     a2*(4*L2 - 1),     b2*(4*L2 - 1), c2*(4*L2 - 1) /)
        B0_tilde(3,:) = (/     a3*(4*L3 - 1),     b3*(4*L3 - 1), c3*(4*L3 - 1) /)
        B0_tilde(4,:) = (/     a4*(4*L4 - 1),     b4*(4*L4 - 1), c4*(4*L4 - 1) /)
        B0_tilde(5,:) = (/ 4*L1*a2 + 4*L2*a1, 4*L1*b2 + 4*L2*b1,  4*L1*c2 + 4*L2*c1 /)
        B0_tilde(6,:) = (/ 4*L2*a3 + 4*L3*a2, 4*L2*b3 + 4*L3*b2,  4*L2*c3 + 4*L3*c2 /)
        B0_tilde(7,:) = (/ 4*L1*a3 + 4*L3*a1, 4*L1*b3 + 4*L3*b1,  4*L1*c3 + 4*L3*c1 /)
        B0_tilde(8,:) = (/ 4*L1*a4 + 4*L4*a1, 4*L1*b4 + 4*L4*b1,  4*L1*c4 + 4*L4*c1 /)
        B0_tilde(9,:) = (/ 4*L2*a4 + 4*L4*a2, 4*L2*b4 + 4*L4*b2,  4*L2*c4 + 4*L4*c2 /)
        B0_tilde(10,:) = (/ 4*L3*a4 + 4*L4*a3, 4*L3*b4 + 4*L4*b3,  4*L3*c4 + 4*L4*c3 /)

        B0_tilde = B0_tilde / det

        H = matmul(transpose(u_e), B0_tilde)
        F = H + EYE
        E = 0.5*(H + transpose(H) + matmul(transpose(H), H))

        call S_Sv_and_C(E, S, S_v, C_SE)
        call compute_b_matrix(B0_tilde, F, B0, 10, 3)

        K_mat = matmul(transpose(B0), matmul(C_SE, B0))
        K_geo_sm = matmul(B0_tilde, matmul(S, transpose(B0_tilde)))

        call scatter_matrix(K_geo_sm, K_geo, 3, 10, 10)

        K = K + (K_mat + K_geo)*det/6 * w
        f_int = f_int + matmul(transpose(B0), S_v) * det/6 * w

        S_exp = S_exp + matmul(reshape(extrapol(:,i), (/10,1/)), &
                 reshape((/S(1,1), S(1,2), S(1,3), S(2,2), S(2,3), S(3,3)/), (/1, 6/)))
        E_exp = E_exp + matmul(reshape(extrapol(:,i), (/10,1/)), &
                 reshape((/ E(1,1), E(1,2), E(1,3), E(2,2), E(2,3), E(3,3)/), (/1, 6/)))

    end do

end subroutine

subroutine hexa8_k_f_s_e(X, u, K, f_int, S_exp, E_exp, S_Sv_and_C)
    implicit none

    real(8), intent(in) :: X(24), u(24)
    real(8), intent(out) :: K(24, 24), f_int(24), S_exp(8,6), E_exp(8,6)
    real(8) :: u_e(8,3), C_SE(6,6)
    real(8) :: K_geo_sm(8,8), K_mat(24,24), K_geo(24,24)
    real(8) :: B0_tilde(8,3), B0(6,24), X_mat(8,3)
    real(8) :: E(3,3), H(3,3), F(3,3), EYE(3,3), S(3,3), S_v(6)
    real(8) :: det, extrapol(8,8), dN_dxi(8,3), dxi_dX(3,3), dX_dxi(3,3)
    real(8) :: gauss_points(8,3), weights(8), a, b, c, d, g, w, xi, eta, zeta
    integer :: i

    ! External functions that will be used afterwards
    external :: scatter_matrix
    external :: compute_b_matrix
    external :: S_Sv_and_C
    external :: invert_3_by_3_matrix


    X_mat = transpose(reshape(X, (/ 3, 8/)))
    u_e = transpose(reshape(u, (/ 3, 8/)))
    EYE = 0.0D0
    EYE(1,1) = 1
    EYE(2,2) = 1
    EYE(3,3) = 1

    a = sqrt(1.0D0/3.0D0)
    gauss_points(1,:) = (/-a,  a,  a /)
    gauss_points(2,:) = (/ a,  a,  a /)
    gauss_points(3,:) = (/-a, -a,  a /)
    gauss_points(4,:) = (/ a, -a,  a /)
    gauss_points(5,:) = (/-a,  a, -a /)
    gauss_points(6,:) = (/ a,  a, -a /)
    gauss_points(7,:) = (/-a, -a, -a /)
    gauss_points(8,:) = (/ a, -a, -a /)

    weights = (/ 1, 1, 1, 1, 1, 1, 1, 1 /)

    b = (-sqrt(3.0D0) + 1)**2*(sqrt(3.0D0) + 1)/8.0D0
    c = (-sqrt(3.0D0) + 1)**3/8.0D0
    d = (sqrt(3.0D0) + 1)**3/8.0D0
    g = (sqrt(3.0D0) + 1)**2*(-sqrt(3.0D0) + 1)/8.0D0
    extrapol(1,:) = (/ b, c, g, b, g, b, d, g /)
    extrapol(2,:) = (/c, b, b, g, b, g, g, d /)
    extrapol(3,:) = (/b, g, c, b, g, d, b, g /)
    extrapol(4,:) = (/g, b, b, c, d, g, g, b /)
    extrapol(5,:) = (/g, b, d, g, b, c, g, b /)
    extrapol(6,:) = (/b, g, g, d, c, b, b, g /)
    extrapol(7,:) = (/g, d, b, g, b, g, c, b /)
    extrapol(8,:) = (/d, g, g, b, g, b, b, c /)


    ! set the matrices and vectors to zero
    K = 0.0
    f_int = 0.0
    S_exp = 0.0
    E_exp = 0.0

    ! Loop over the gauss points
    do i = 1, 8
        xi = gauss_points(i, 1)
        eta = gauss_points(i, 2)
        zeta = gauss_points(i, 3)
        w  = weights(i)

        dN_dxi(1,:) = (/-(-eta+1)*(-zeta+1), -(-xi+1)*(-zeta+1), -(-eta+1)*(-xi+1) /)
        dN_dxi(2,:) = (/ (-eta+1)*(-zeta+1),  -(xi+1)*(-zeta+1),  -(-eta+1)*(xi+1) /)
        dN_dxi(3,:) = (/  (eta+1)*(-zeta+1),   (xi+1)*(-zeta+1),   -(eta+1)*(xi+1) /)
        dN_dxi(4,:) = (/ -(eta+1)*(-zeta+1),  (-xi+1)*(-zeta+1),  -(eta+1)*(-xi+1) /)
        dN_dxi(5,:) = (/ -(-eta+1)*(zeta+1),  -(-xi+1)*(zeta+1),  (-eta+1)*(-xi+1) /)
        dN_dxi(6,:) = (/  (-eta+1)*(zeta+1),   -(xi+1)*(zeta+1),   (-eta+1)*(xi+1) /)
        dN_dxi(7,:) = (/   (eta+1)*(zeta+1),    (xi+1)*(zeta+1),    (eta+1)*(xi+1) /)
        dN_dxi(8,:) = (/  -(eta+1)*(zeta+1),   (-xi+1)*(zeta+1),   (eta+1)*(-xi+1) /)
        dN_dxi = dN_dxi / 8.0D0

        dX_dxi = matmul(transpose(X_mat), dN_dxi)
        call invert_3_by_3_matrix(dX_dxi, dxi_dX, det)
        B0_tilde = matmul(dN_dxi, dxi_dX)

        H = matmul(transpose(u_e), B0_tilde)
        F = H + EYE
        E = 0.5*(H + transpose(H) + matmul(transpose(H), H))

        call S_Sv_and_C(E, S, S_v, C_SE)
        call compute_b_matrix(B0_tilde, F, B0, 8, 3)

        K_mat = matmul(transpose(B0), matmul(C_SE, B0))
        K_geo_sm = matmul(B0_tilde, matmul(S, transpose(B0_tilde)))

        call scatter_matrix(K_geo_sm, K_geo, 3, 8, 8)

        K = K + (K_mat + K_geo)*det * w
        f_int = f_int + matmul(transpose(B0), S_v) * det * w

        S_exp = S_exp + matmul(reshape(extrapol(:,i), (/8,1/)), &
                 reshape((/S(1,1), S(1,2), S(1,3), S(2,2), S(2,3), S(3,3)/), (/1, 6/)))
        E_exp = E_exp + matmul(reshape(extrapol(:,i), (/8,1/)), &
                 reshape((/ E(1,1), E(1,2), E(1,3), E(2,2), E(2,3), E(3,3)/), (/1, 6/)))

    end do
end subroutine


subroutine hexa20_k_f_s_e(X, u, K, f_int, S_exp, E_exp, S_Sv_and_C)
        implicit none

        real(8), intent(in) :: X(60), u(60)
        real(8), intent(out) :: K(60, 60), f_int(60), S_exp(20,6), E_exp(20,6)
        real(8) :: u_e(20,3), C_SE(6,6)
        real(8) :: K_geo_sm(20,20), K_mat(60,60), K_geo(60,60)
        real(8) :: B0_tilde(20,3), B0(6,60), X_mat(20,3)
        real(8) :: E(3,3), H(3,3), F(3,3), EYE(3,3), S(3,3), S_v(6)
        real(8) :: det, extrapol(20,27), dN_dxi(20,3), dxi_dX(3,3), dX_dxi(3,3)
        real(8) :: gauss_points(27,3), weights(27), a, w, wa, w0, xi, eta, zeta
        real(8) :: b1, b2, b3, b4, b5, b6, b7, b8, b9, c1, c2, c3, c4, c5, c6
        integer :: i

        ! External functions that will be used afterwards
        external :: scatter_matrix
        external :: compute_b_matrix
        external :: S_Sv_and_C
        external :: invert_3_by_3_matrix


        X_mat = transpose(reshape(X, (/ 3, 20/)))
        u_e = transpose(reshape(u, (/ 3, 20/)))
        EYE = 0.0D0
        EYE(1,1) = 1
        EYE(2,2) = 1
        EYE(3,3) = 1

        a = sqrt(3.0D0/5.0D0)
        wa = 5/9.0D0
        w0 = 8/9.0D0
        gauss_points(1,:) = (/ -a, -a, -a /)
        gauss_points(2,:) = (/ 0.0D0 , -a, -a /)
        gauss_points(3,:) = (/  a, -a, -a /)
        gauss_points(4,:) = (/ -a, 0.0D0 , -a /)
        gauss_points(5,:) = (/ 0.0D0 , 0.0D0 , -a /)
        gauss_points(6,:) = (/  a, 0.0D0 , -a /)
        gauss_points(7,:) = (/ -a,  a, -a /)
        gauss_points(8,:) = (/ 0.0D0 ,  a, -a /)
        gauss_points(9,:) = (/  a,  a, -a /)
        gauss_points(10,:) = (/ -a, -a, 0.0D0  /)
        gauss_points(11,:) = (/ 0.0D0 , -a, 0.0D0  /)
        gauss_points(12,:) = (/  a, -a, 0.0D0  /)
        gauss_points(13,:) = (/ -a, 0.0D0 , 0.0D0  /)
        gauss_points(14,:) = (/ 0.0D0 , 0.0D0 , 0.0D0  /)
        gauss_points(15,:) = (/  a, 0.0D0 , 0.0D0  /)
        gauss_points(16,:) = (/ -a,  a, 0.0D0  /)
        gauss_points(17,:) = (/ 0.0D0 ,  a, 0.0D0  /)
        gauss_points(18,:) = (/  a,  a, 0.0D0  /)
        gauss_points(19,:) = (/ -a, -a,  a /)
        gauss_points(20,:) = (/ 0.0D0 , -a,  a /)
        gauss_points(21,:) = (/  a, -a,  a /)
        gauss_points(22,:) = (/ -a, 0.0D0 ,  a /)
        gauss_points(23,:) = (/ 0.0D0 , 0.0D0 ,  a /)
        gauss_points(24,:) = (/  a, 0.0D0 ,  a /)
        gauss_points(25,:) = (/ -a,  a,  a /)
        gauss_points(26,:) = (/ 0.0D0 ,  a,  a /)
        gauss_points(27,:) = (/  a,  a,  a /)
        weights = (/ wa*wa*wa, w0*wa*wa, wa*wa*wa, wa*w0*wa, w0*w0*wa, &
                     wa*w0*wa, wa*wa*wa, w0*wa*wa, wa*wa*wa, wa*wa*w0, &
                     w0*wa*w0, wa*wa*w0, wa*w0*w0, w0*w0*w0, wa*w0*w0, &
                     wa*wa*w0, w0*wa*w0, wa*wa*w0, wa*wa*wa, w0*wa*wa, &
                     wa*wa*wa, wa*w0*wa, w0*w0*wa, wa*w0*wa, wa*wa*wa, &
                     w0*wa*wa, wa*wa*wa /)

         b1 = 13.0D0*sqrt(15.0D0)/36.0D0 + 17.0D0/12.0D0
         b2 = (4.0D0 + sqrt(15.0D0))/9.0D0
         b3 = (1.0D0 + sqrt(15.0D0))/36.0D0
         b4 = (3.0D0 + sqrt(15.0D0))/27.0D0
         b5 = 1.0D0/9.0D0
         b6 = (1.0D0 - sqrt(15.0D0))/36.0D0
         b7 = -2.0D0/27.0D0
         b8 = (3.0D0 - sqrt(15.0D0))/27.0D0
         b9 = -13*sqrt(15.0D0)/36 + 17/12.0D0
         c1 = (-4.0D0 + sqrt(15.0D0))/9.0D0
         c2 = (3.0D0 + sqrt(15.0D0))/18.0D0
         c3 = sqrt(15.0D0)/6.0D0 + 2/3.0D0
         c4 = 3.0D0/18.0D0
         c5 = (- 3.0D0 + sqrt(15.0D0))/18.0D0
         c6 = (4.0D0 - sqrt(15.0D0))/6.0D0

        extrapol(1,:) = (/ b1,-b2,b3,-b2,b4,b5,b3,b5,b6,-b2,b4,b5,b4,b7,b8,b5,b8,c1,b3,b5,b6,b5,b8,c1,b6,c1,b9 /)
        extrapol(2,:) = (/ b3,-b2,b1,b5,b4,-b2,b6,b5,b3,b5,b4,-b2,b8,b7,b4,c1,b8,b5,b6,b5,b3,c1,b8,b5,b9,c1,b6 /)
        extrapol(3,:) = (/ b6,b5,b3,b5,b4,-b2,b3,-b2,b1,c1,b8,b5,b8,b7,b4,b5,b4,-b2,b9,c1,b6,c1,b8,b5,b6,b5,b3 /)
        extrapol(4,:) = (/ b3,b5,b6,-b2,b4,b5,b1,-b2,b3,b5,b8,c1,b4,b7,b8,-b2,b4,b5,b6,c1,b9,b5,b8,c1,b3,b5,b6 /)
        extrapol(5,:) = (/ b3,b5,b6,b5,b8,c1,b6,c1,b9,-b2,b4,b5,b4,b7,b8,b5,b8,c1,b1,-b2,b3,-b2,b4,b5,b3,b5,b6 /)
        extrapol(6,:) = (/ b6,b5,b3,c1,b8,b5,b9,c1,b6,b5,b4,-b2,b8,b7,b4,c1,b8,b5,b3,-b2,b1,b5,b4,-b2,b6,b5,b3 /)
        extrapol(7,:) = (/ b9,c1,b6,c1,b8,b5,b6,b5,b3,c1,b8,b5,b8,b7,b4,b5,b4,-b2,b6,b5,b3,b5,b4,-b2,b3,-b2,b1 /)
        extrapol(8,:) = (/ b6,c1,b9,b5,b8,c1,b3,b5,b6,b5,b8,c1,b4,b7,b8,-b2,b4,b5,b3,b5,b6,-b2,b4,b5,b1,-b2,b3 /)
        extrapol(9,:) = (/ c2,c3,c2,-c2,-c2,-c2,c4,-c4,c4,-c2,-c2,-c2,b5,b5,b5,c5,c5,c5,c4,-c4,c4,c5,c5,c5,-c5,c6,-c5 /)
        extrapol(10,:) = (/ c4,-c2,c2,-c4,-c2,c3,c4,-c2,c2,c5,b5,-c2,c5,b5,-c2,c5,b5,-c2,-c5,c5,c4,c6,c5,-c4,-c5,c5,c4 /)
        extrapol(11,:) = (/ c4,-c4,c4,-c2,-c2,-c2,c2,c3,c2,c5,c5,c5,b5,b5,b5,-c2,-c2,-c2,-c5,c6,-c5,c5,c5,c5,c4,-c4,c4 /)
        extrapol(12,:) = (/ c2,-c2,c4,c3,-c2,-c4,c2,-c2,c4,-c2,b5,c5,-c2,b5,c5,-c2,b5,c5,c4,c5,-c5,-c4,c5,c6,c4,c5,-c5 /)
        extrapol(13,:) = (/ c4,-c4,c4,c5,c5,c5,-c5,c6,-c5,-c2,-c2,-c2,b5,b5,b5,c5,c5,c5,c2,c3,c2,-c2,-c2,-c2,c4,-c4,c4 /)
        extrapol(14,:) = (/ -c5,c5,c4,c6,c5,-c4,-c5,c5,c4,c5,b5,-c2,c5,b5,-c2,c5,b5,-c2,c4,-c2,c2,-c4,-c2,c3,c4,-c2,c2 /)
        extrapol(15,:) = (/ -c5,c6,-c5,c5,c5,c5,c4,-c4,c4,c5,c5,c5,b5,b5,b5,-c2,-c2,-c2,c4,-c4,c4,-c2,-c2,-c2,c2,c3,c2 /)
        extrapol(16,:) = (/ c4,c5,-c5,-c4,c5,c6,c4,c5,-c5,-c2,b5,c5,-c2,b5,c5,-c2,b5,c5,c2,-c2,c4,c3,-c2,-c4,c2,-c2,c4 /)
        extrapol(17,:) = (/ c2,-c2,c4,-c2,b5,c5,c4,c5,-c5,c3,-c2,-c4,-c2,b5,c5,-c4,c5,c6,c2,-c2,c4,-c2,b5,c5,c4,c5,-c5 /)
        extrapol(18,:) = (/ c4,-c2,c2,c5,b5,-c2,-c5,c5,c4,-c4,-c2,c3,c5,b5,-c2,c6,c5,-c4,c4,-c2,c2,c5,b5,-c2,-c5,c5,c4 /)
        extrapol(19,:) = (/ -c5,c5,c4,c5,b5,-c2,c4,-c2,c2,c6,c5,-c4,c5,b5,-c2,-c4,-c2,c3,-c5,c5,c4,c5,b5,-c2,c4,-c2,c2 /)
        extrapol(20,:) = (/ c4,c5,-c5,-c2,b5,c5,c2,-c2,c4,-c4,c5,c6,-c2,b5,c5,c3,-c2,-c4,c4,c5,-c5,-c2,b5,c5,c2,-c2,c4 /)

        ! set the matrices and vectors to zero
        K = 0.0
        f_int = 0.0
        S_exp = 0.0
        E_exp = 0.0

        ! Loop over the gauss points
        do i = 1, 27
            xi = gauss_points(i, 1)
            eta = gauss_points(i, 2)
            zeta = gauss_points(i, 3)
            w  = weights(i)

            dN_dxi(1,:) = (/ (eta-1)*(zeta-1)*(eta+2*xi+zeta+1), &
                             (xi-1)*(zeta-1)*(2*eta+xi+zeta+1), &
                             (eta-1)*(xi-1)*(eta+xi+2*zeta+1) /)
            dN_dxi(2,:) = (/ (eta-1)*(zeta-1)*(-eta+2*xi-zeta-1), &
                             (xi+1)*(zeta-1)*(-2*eta+xi-zeta-1), &
                             (eta-1)*(xi+1)*(-eta+xi-2*zeta-1) /)
            dN_dxi(3,:) = (/ (eta+1)*(zeta-1)*(-eta-2*xi+zeta+1), &
                             (xi+1)*(zeta-1)*(-2*eta-xi+zeta+1), &
                             (eta+1)*(xi+1)*(-eta-xi+2*zeta+1) /)
            dN_dxi(4,:) = (/ (eta+1)*(zeta-1)*(eta-2*xi-zeta-1), &
                             (xi-1)*(zeta-1)*(2*eta-xi-zeta-1), &
                             (eta+1)*(xi-1)*(eta-xi-2*zeta-1) /)
            dN_dxi(5,:) = (/(eta-1)*(zeta+1)*(-eta-2*xi+zeta-1), &
                            (xi-1)*(zeta+1)*(-2*eta-xi+zeta-1), &
                            (eta-1)*(xi-1)*(-eta-xi+2*zeta-1) /)
            dN_dxi(6,:) = (/ (eta-1)*(zeta+1)*(eta-2*xi-zeta+1), &
                             (xi+1)*(zeta+1)*(2*eta-xi-zeta+1), &
                             (eta-1)*(xi+1)*(eta-xi-2*zeta+1) /)
            dN_dxi(7,:) = (/ (eta+1)*(zeta+1)*(eta+2*xi+zeta-1), &
                             (xi+1)*(zeta+1)*(2*eta+xi+zeta-1), &
                             (eta+1)*(xi+1)*(eta+xi+2*zeta-1) /)
            dN_dxi(8,:) = (/(eta+1)*(zeta+1)*(-eta+2*xi-zeta+1), &
                            (xi-1)*(zeta+1)*(-2*eta+xi-zeta+1), &
                            (eta+1)*(xi-1)*(-eta+xi-2*zeta+1) /)
            dN_dxi(9,:) = (/-4*xi*(eta-1)*(zeta-1), -2*(xi**2-1)*(zeta-1), &
                            -2*(eta-1)*(xi**2-1) /)
            dN_dxi(10,:) = (/ 2*(eta**2-1)*(zeta-1), 4*eta*(xi+1)*(zeta-1), &
                              2*(eta**2-1)*(xi+1) /)
            dN_dxi(11,:) = (/ 4*xi*(eta+1)*(zeta-1),  2*(xi**2-1)*(zeta-1), &
                              2*(eta+1)*(xi**2-1) /)
            dN_dxi(12,:) = (/-2*(eta**2-1)*(zeta-1),-4*eta*(xi-1)*(zeta-1), &
                             -2*(eta**2-1)*(xi-1) /)
            dN_dxi(13,:) = (/ 4*xi*(eta-1)*(zeta+1),  2*(xi**2-1)*(zeta+1), &
                              2*(eta-1)*(xi**2-1) /)
            dN_dxi(14,:) = (/-2*(eta**2-1)*(zeta+1),-4*eta*(xi+1)*(zeta+1), &
                             -2*(eta**2-1)*(xi+1) /)
            dN_dxi(15,:) = (/-4*xi*(eta+1)*(zeta+1), -2*(xi**2-1)*(zeta+1), &
                             -2*(eta+1)*(xi**2-1) /)
            dN_dxi(16,:) = (/ 2*(eta**2-1)*(zeta+1), 4*eta*(xi-1)*(zeta+1), &
                              2*(eta**2-1)*(xi-1) /)
            dN_dxi(17,:) = (/-2*(eta-1)*(zeta**2-1), -2*(xi-1)*(zeta**2-1), &
                             -4*zeta*(eta-1)*(xi-1) /)
            dN_dxi(18,:) = (/ 2*(eta-1)*(zeta**2-1),  2*(xi+1)*(zeta**2-1), &
                              4*zeta*(eta-1)*(xi+1) /)
            dN_dxi(19,:) = (/-2*(eta+1)*(zeta**2-1), -2*(xi+1)*(zeta**2-1), &
                             -4*zeta*(eta+1)*(xi+1) /)
            dN_dxi(20,:) = (/ 2*(eta+1)*(zeta**2-1),  2*(xi-1)*(zeta**2-1), &
                              4*zeta*(eta+1)*(xi-1) /)

            dN_dxi = dN_dxi / 8.0D0

            dX_dxi = matmul(transpose(X_mat), dN_dxi)
            call invert_3_by_3_matrix(dX_dxi, dxi_dX, det)
            B0_tilde = matmul(dN_dxi, dxi_dX)

            H = matmul(transpose(u_e), B0_tilde)
            F = H + EYE
            E = 0.5*(H + transpose(H) + matmul(transpose(H), H))

            call S_Sv_and_C(E, S, S_v, C_SE)
            call compute_b_matrix(B0_tilde, F, B0, 20, 3)

            K_mat = matmul(transpose(B0), matmul(C_SE, B0))
            K_geo_sm = matmul(B0_tilde, matmul(S, transpose(B0_tilde)))

            call scatter_matrix(K_geo_sm, K_geo, 3, 20, 20)

            K = K + (K_mat + K_geo)*det * w
            f_int = f_int + matmul(transpose(B0), S_v) * det * w

            S_exp = S_exp + matmul(reshape(extrapol(:,i), (/20,1/)), &
                     reshape((/S(1,1), S(1,2), S(1,3), S(2,2), S(2,3), S(3,3)/), (/1, 6/)))
            E_exp = E_exp + matmul(reshape(extrapol(:,i), (/20,1/)), &
                     reshape((/ E(1,1), E(1,2), E(1,3), E(2,2), E(2,3), E(3,3)/), (/1, 6/)))

        end do
end subroutine

subroutine hexa20_m(X, rho, M)
    implicit none

    real(8), intent(in) :: X(60), rho
    real(8), intent(out) :: M(60, 60)
    real(8) :: gauss_points(27,3), weights(27), xi, eta, zeta
    real(8) :: dX_dxi(3,3), dN_dxi(20,3), X_mat(20,3), det
    real(8) :: N(20,1), M_small(20,20)
    real(8) :: a, w, wa, w0
    integer :: i

    external :: scatter_matrix

    X_mat = transpose(reshape(X, (/ 3, 20/)))

    a = sqrt(3.0D0/5.0D0)
    wa = 5/9.0D0
    w0 = 8/9.0D0
    gauss_points(1,:) = (/ -a, -a, -a /)
    gauss_points(2,:) = (/ 0.0D0 , -a, -a /)
    gauss_points(3,:) = (/  a, -a, -a /)
    gauss_points(4,:) = (/ -a, 0.0D0 , -a /)
    gauss_points(5,:) = (/ 0.0D0 , 0.0D0 , -a /)
    gauss_points(6,:) = (/  a, 0.0D0 , -a /)
    gauss_points(7,:) = (/ -a,  a, -a /)
    gauss_points(8,:) = (/ 0.0D0 ,  a, -a /)
    gauss_points(9,:) = (/  a,  a, -a /)
    gauss_points(10,:) = (/ -a, -a, 0.0D0  /)
    gauss_points(11,:) = (/ 0.0D0 , -a, 0.0D0  /)
    gauss_points(12,:) = (/  a, -a, 0.0D0  /)
    gauss_points(13,:) = (/ -a, 0.0D0 , 0.0D0  /)
    gauss_points(14,:) = (/ 0.0D0 , 0.0D0 , 0.0D0  /)
    gauss_points(15,:) = (/  a, 0.0D0 , 0.0D0  /)
    gauss_points(16,:) = (/ -a,  a, 0.0D0  /)
    gauss_points(17,:) = (/ 0.0D0 ,  a, 0.0D0  /)
    gauss_points(18,:) = (/  a,  a, 0.0D0  /)
    gauss_points(19,:) = (/ -a, -a,  a /)
    gauss_points(20,:) = (/ 0.0D0 , -a,  a /)
    gauss_points(21,:) = (/  a, -a,  a /)
    gauss_points(22,:) = (/ -a, 0.0D0 ,  a /)
    gauss_points(23,:) = (/ 0.0D0 , 0.0D0 ,  a /)
    gauss_points(24,:) = (/  a, 0.0D0 ,  a /)
    gauss_points(25,:) = (/ -a,  a,  a /)
    gauss_points(26,:) = (/ 0.0D0 ,  a,  a /)
    gauss_points(27,:) = (/  a,  a,  a /)
    weights = (/ wa*wa*wa, w0*wa*wa, wa*wa*wa, wa*w0*wa, w0*w0*wa, &
                 wa*w0*wa, wa*wa*wa, w0*wa*wa, wa*wa*wa, wa*wa*w0, &
                 w0*wa*w0, wa*wa*w0, wa*w0*w0, w0*w0*w0, wa*w0*w0, &
                 wa*wa*w0, w0*wa*w0, wa*wa*w0, wa*wa*wa, w0*wa*wa, &
                 wa*wa*wa, wa*w0*wa, w0*w0*wa, wa*w0*wa, wa*wa*wa, &
                 w0*wa*wa, wa*wa*wa /)

    M = 0
    M_small = 0

    ! Loop over the gauss points
    do i = 1, 27
        xi = gauss_points(i, 1)
        eta = gauss_points(i, 2)
        zeta = gauss_points(i, 3)
        w  = weights(i)

        dN_dxi(1,:) = (/ (eta-1)*(zeta-1)*(eta+2*xi+zeta+1), &
                         (xi-1)*(zeta-1)*(2*eta+xi+zeta+1), &
                         (eta-1)*(xi-1)*(eta+xi+2*zeta+1) /)
        dN_dxi(2,:) = (/ (eta-1)*(zeta-1)*(-eta+2*xi-zeta-1), &
                         (xi+1)*(zeta-1)*(-2*eta+xi-zeta-1), &
                         (eta-1)*(xi+1)*(-eta+xi-2*zeta-1) /)
        dN_dxi(3,:) = (/ (eta+1)*(zeta-1)*(-eta-2*xi+zeta+1), &
                         (xi+1)*(zeta-1)*(-2*eta-xi+zeta+1), &
                         (eta+1)*(xi+1)*(-eta-xi+2*zeta+1) /)
        dN_dxi(4,:) = (/ (eta+1)*(zeta-1)*(eta-2*xi-zeta-1), &
                         (xi-1)*(zeta-1)*(2*eta-xi-zeta-1), &
                         (eta+1)*(xi-1)*(eta-xi-2*zeta-1) /)
        dN_dxi(5,:) = (/(eta-1)*(zeta+1)*(-eta-2*xi+zeta-1), &
                        (xi-1)*(zeta+1)*(-2*eta-xi+zeta-1), &
                        (eta-1)*(xi-1)*(-eta-xi+2*zeta-1) /)
        dN_dxi(6,:) = (/ (eta-1)*(zeta+1)*(eta-2*xi-zeta+1), &
                         (xi+1)*(zeta+1)*(2*eta-xi-zeta+1), &
                         (eta-1)*(xi+1)*(eta-xi-2*zeta+1) /)
        dN_dxi(7,:) = (/ (eta+1)*(zeta+1)*(eta+2*xi+zeta-1), &
                         (xi+1)*(zeta+1)*(2*eta+xi+zeta-1), &
                         (eta+1)*(xi+1)*(eta+xi+2*zeta-1) /)
        dN_dxi(8,:) = (/(eta+1)*(zeta+1)*(-eta+2*xi-zeta+1), &
                        (xi-1)*(zeta+1)*(-2*eta+xi-zeta+1), &
                        (eta+1)*(xi-1)*(-eta+xi-2*zeta+1) /)
        dN_dxi(9,:) = (/-4*xi*(eta-1)*(zeta-1), -2*(xi**2-1)*(zeta-1), &
                        -2*(eta-1)*(xi**2-1) /)
        dN_dxi(10,:) = (/ 2*(eta**2-1)*(zeta-1), 4*eta*(xi+1)*(zeta-1), &
                          2*(eta**2-1)*(xi+1) /)
        dN_dxi(11,:) = (/ 4*xi*(eta+1)*(zeta-1),  2*(xi**2-1)*(zeta-1), &
                          2*(eta+1)*(xi**2-1) /)
        dN_dxi(12,:) = (/-2*(eta**2-1)*(zeta-1),-4*eta*(xi-1)*(zeta-1), &
                         -2*(eta**2-1)*(xi-1) /)
        dN_dxi(13,:) = (/ 4*xi*(eta-1)*(zeta+1),  2*(xi**2-1)*(zeta+1), &
                          2*(eta-1)*(xi**2-1) /)
        dN_dxi(14,:) = (/-2*(eta**2-1)*(zeta+1),-4*eta*(xi+1)*(zeta+1), &
                         -2*(eta**2-1)*(xi+1) /)
        dN_dxi(15,:) = (/-4*xi*(eta+1)*(zeta+1), -2*(xi**2-1)*(zeta+1), &
                         -2*(eta+1)*(xi**2-1) /)
        dN_dxi(16,:) = (/ 2*(eta**2-1)*(zeta+1), 4*eta*(xi-1)*(zeta+1), &
                          2*(eta**2-1)*(xi-1) /)
        dN_dxi(17,:) = (/-2*(eta-1)*(zeta**2-1), -2*(xi-1)*(zeta**2-1), &
                         -4*zeta*(eta-1)*(xi-1) /)
        dN_dxi(18,:) = (/ 2*(eta-1)*(zeta**2-1),  2*(xi+1)*(zeta**2-1), &
                          4*zeta*(eta-1)*(xi+1) /)
        dN_dxi(19,:) = (/-2*(eta+1)*(zeta**2-1), -2*(xi+1)*(zeta**2-1), &
                         -4*zeta*(eta+1)*(xi+1) /)
        dN_dxi(20,:) = (/ 2*(eta+1)*(zeta**2-1),  2*(xi-1)*(zeta**2-1), &
                          4*zeta*(eta+1)*(xi-1) /)

        dN_dxi = dN_dxi / 8.0D0

        dX_dxi = matmul(transpose(X_mat), dN_dxi)
        det =   dX_dxi(1,1)*dX_dxi(2,2)*dX_dxi(3,3) &
              - dX_dxi(1,1)*dX_dxi(2,3)*dX_dxi(3,2) &
              - dX_dxi(1,2)*dX_dxi(2,1)*dX_dxi(3,3) &
              + dX_dxi(1,2)*dX_dxi(2,3)*dX_dxi(3,1) &
              + dX_dxi(1,3)*dX_dxi(2,1)*dX_dxi(3,2) &
              - dX_dxi(1,3)*dX_dxi(2,2)*dX_dxi(3,1)

       N(:,1) = (/ (eta-1)*(xi-1)*(zeta-1)*(eta+xi+zeta+2), &
                  -(eta-1)*(xi+1)*(zeta-1)*(eta-xi+zeta+2), &
                  -(eta+1)*(xi+1)*(zeta-1)*(eta+xi-zeta-2), &
                 -(eta+1)*(xi-1)*(zeta-1)*(-eta+xi+zeta+2), &
                  -(eta-1)*(xi-1)*(zeta+1)*(eta+xi-zeta+2), &
                   (eta-1)*(xi+1)*(zeta+1)*(eta-xi-zeta+2), &
                   (eta+1)*(xi+1)*(zeta+1)*(eta+xi+zeta-2), &
                  -(eta+1)*(xi-1)*(zeta+1)*(eta-xi+zeta-2), &
                             -2*(eta-1)*(xi**2-1)*(zeta-1), &
                              2*(eta**2-1)*(xi+1)*(zeta-1), &
                              2*(eta+1)*(xi**2-1)*(zeta-1), &
                             -2*(eta**2-1)*(xi-1)*(zeta-1), &
                              2*(eta-1)*(xi**2-1)*(zeta+1), &
                             -2*(eta**2-1)*(xi+1)*(zeta+1), &
                             -2*(eta+1)*(xi**2-1)*(zeta+1), &
                              2*(eta**2-1)*(xi-1)*(zeta+1), &
                             -2*(eta-1)*(xi-1)*(zeta**2-1), &
                              2*(eta-1)*(xi+1)*(zeta**2-1), &
                             -2*(eta+1)*(xi+1)*(zeta**2-1), &
                              2*(eta+1)*(xi-1)*(zeta**2-1) /)

        N = N / 8.0D0
        M_small = M_small + matmul(N, transpose(N)) * det * rho * w
    end do

    call scatter_matrix(M_small, M, 3, 20, 20)
end subroutine
