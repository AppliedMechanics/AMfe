

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


subroutine tri3_k_and_f(X, u, K, f_int, t, S_Sv_and_C_2d)
    implicit none

!   What is important here is that the
    real(8), intent(in) :: X(6), u(6), t
    real(8), intent(out) :: K(6, 6), f_int(6)
    real(8) :: X1, X2, X3, Y1, Y2, Y3, A0
    real(8) :: u_e(3,2), C_SE(3,3)
    real(8) :: K_geo_sm(3,3), K_mat(6,6), K_geo(6,6)
    real(8) :: B0_tilde(3,2),  B0(3,6)
    real(8) :: E(2,2), H(2,2), F(2,2), EYE(2,2), S(2,2), S_v(3)
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

end subroutine tri3_k_and_f


subroutine tri6_k_and_f(X, u, K, f_int, t, S_Sv_and_C_2d)
    implicit none

    integer :: i
    real(8), intent(in) :: X(12), u(12), t
    real(8), intent(out) :: K(12, 12), f_int(12)
    real(8) :: X1, X2, X3, Y1, Y2, Y3, X4, Y4, X5, Y5, X6, Y6, A0
    real(8) :: Jx1 ,Jx2 ,Jx3 ,Jy1 ,Jy2 ,Jy3, w
    real(8) :: u_e(6,2), C_SE(3,3)
    real(8) :: K_geo_sm(6,6), K_mat(12,12), K_geo(12,12)
    real(8) :: B0_tilde(6,2), B0(3,12)
    real(8) :: E(2,2), H(2,2), F(2,2), EYE(2,2), S(2,2), S_v(3)

    real(8) :: gauss_points(7,3), weights(7)
    real(8) :: w0, w1, w2, alpha1, alpha2, beta1, beta2
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

    ! take care of a precise description of 1/3 in order to avoid errors!
    w0 = 0.225D0

    alpha1 = 0.0597158717D0
    beta1 = 0.4701420641D0
    w1 = 0.1323941527D0

    alpha2 = 0.7974269853D0
    beta2 = 0.1012865073D0
    w2 = 0.1259391805D0

    weights = (/ w0, w1, w1, w1, w2, w2, w2 /)
    gauss_points(1,:) = (/ 1/3.0D0, 1/3.0D0, 1/3.0D0 /)
    gauss_points(2,:) = (/ alpha1, beta1, beta1 /)
    gauss_points(3,:) = (/ beta1, alpha1, beta1 /)
    gauss_points(4,:) = (/ beta1, beta1, alpha1 /)
    gauss_points(5,:) = (/ alpha2, beta2, beta2 /)
    gauss_points(6,:) = (/ beta2, alpha2, beta2 /)
    gauss_points(7,:) = (/ beta2, beta2, alpha2 /)


    EYE = reshape((/1, 0, 0, 1/), shape(EYE))

    K = 0.0
    f_int = 0.0

    ! loop over all quadrature points
    do i=1,7
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

    end do
end subroutine


subroutine tri6_m(X, rho, t, M)
    implicit none

    integer :: i
    real(8), intent(in) :: X(12), t, rho
    real(8), intent(out) :: M(12,12)
    real(8) :: X1, X2, X3, Y1, Y2, Y3, X4, Y4, X5, Y5, X6, Y6
    real(8) :: Jx1 ,Jx2 ,Jx3 ,Jy1 ,Jy2 ,Jy3, w
    real(8) :: N(6,1), M_small(6,6)

    real(8) :: gauss_points(7,3), weights(7)
    real(8) :: w0, w1, w2, alpha1, alpha2, beta1, beta2
    real(8) :: L1, L2, L3, det

!   External functions that will be used afterwards
    external :: scatter_matrix

    ! take care of a precise description of 1/3 in order to avoid errors!
    w0 = 0.225D0

    alpha1 = 0.0597158717D0
    beta1 = 0.4701420641D0
    w1 = 0.1323941527D0

    alpha2 = 0.7974269853D0
    beta2 = 0.1012865073D0
    w2 = 0.1259391805D0

    weights = (/ w0, w1, w1, w1, w2, w2, w2 /)
    gauss_points(1,:) = (/ 1/3.0D0, 1/3.0D0, 1/3.0D0 /)
    gauss_points(2,:) = (/ alpha1, beta1, beta1 /)
    gauss_points(3,:) = (/ beta1, alpha1, beta1 /)
    gauss_points(4,:) = (/ beta1, beta1, alpha1 /)
    gauss_points(5,:) = (/ alpha2, beta2, beta2 /)
    gauss_points(6,:) = (/ beta2, alpha2, beta2 /)
    gauss_points(7,:) = (/ beta2, beta2, alpha2 /)


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

    do i=1,7
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


subroutine tet4_k_and_f(X, u, K, f_int, S_Sv_and_C)
    implicit none

    real(8), intent(in) :: X(12), u(12)
    real(8), intent(out) :: K(12, 12), f_int(12)
    real(8) :: X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4
    real(8) :: u_e(4,3), C_SE(6,6)
    real(8) :: K_geo_sm(4,4), K_mat(12,12), K_geo(12,12)
    real(8) :: B0_tilde(4,3), B0(6,12)
    real(8) :: E(3,3), H(3,3), F(3,3), EYE(3,3), S(3,3), S_v(6)
    real(8) :: det

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

end subroutine


subroutine tet10_k_and_f(X, u, K, f_int, S_Sv_and_C)
    implicit none

    real(8), intent(in) :: X(30), u(30)
    real(8), intent(out) :: K(30, 30), f_int(30)
    real(8) :: X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4, X5, Y5, Z5
    real(8) :: X6, Y6, Z6, X7, Y7, Z7, X8, Y8, Z8, X9, Y9, Z9, X10, Y10, Z10
    real(8) :: Jx1, Jx2, Jx3, Jx4, Jy1, Jy2, Jy3, Jy4, Jz1, Jz2, Jz3, Jz4
    real(8) :: a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4
    real(8) :: L1, L2, L3, L4
    real(8) :: u_e(10,3), C_SE(6,6)
    real(8) :: K_geo_sm(10,10), K_mat(30,30), K_geo(30,30)
    real(8) :: B0_tilde(10,3), B0(6,30)
    real(8) :: E(3,3), H(3,3), F(3,3), EYE(3,3), S(3,3), S_v(6)
    real(8) :: det
    real(8) :: gauss_points(4,4), weights(4), a, b, w
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

    gauss_points(1,:) = (/ a, a, a, b /)
    gauss_points(2,:) = (/ a, a, b, a /)
    gauss_points(3,:) = (/ a, b, a, a /)
    gauss_points(4,:) = (/ b, a, a, a /)

    weights = (/ w, w, w, w /)

    ! set the matrices and vectors to zero
    K = 0.0
    f_int = 0.0

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

    end do

end subroutine
