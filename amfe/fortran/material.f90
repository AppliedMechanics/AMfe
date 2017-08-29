! Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
! Universitaet Muenchen.
!
! Distributed under BSD-3-Clause License. See LICENSE-File for more information
!
subroutine mooney_rivlin_S_Sv_and_C(E, S, Sv, C_SE, A10, A01, kappa)
      implicit none
      real(8), intent(in) :: E(3,3), A10, A01, kappa
      real(8), intent(out) :: S(3,3), Sv(6), C_SE(6,6)
      real(8) :: C(3,3), I1E(6,1), I2E(6,1), I3E(6,1), I2EE(6,6), I3EE(6,6), &
                 J1E(6,1), J2E(6,1), J3E(6,1), EYE(3,3), J1EE(6,6), J2EE(6,6), J3EE(6,6)
      real(8) :: C11, C22, C33, C23, C13, C12, I1, I2, I3, J3, &
                 J1I1, J1I3, J2I2, J2I3, J3I3, J1I1I3, J1I3I3, J2I2I3, J2I3I3, J3I3I3

      EYE = reshape((/1, 0, 0, 0, 1, 0, 0, 0, 1/), shape(EYE))
      C = 2*E + EYE

      C11 = C(1,1)
      C22 = C(2,2)
      C33 = C(3,3)
      C23 = C(2,3)
      C13 = C(1,3)
      C12 = C(1,2)
!     invariants and reduced invariants
      I1  = C11 + C22 + C33
      I2  = C11*C22 + C11*C33 - C12**2 - C13**2 + C22*C33 - C23**2
      I3  = C11*C22*C33 - C11*C23**2 - C12**2*C33 + 2*C12*C13*C23 - C13**2*C22

      J3  = sqrt(I3)
!     derivatives
      J1I1 = I3**(-1.0/3.0D0)
      J1I3 = -I1/(3*I3**(4.0D0/3.00D0))
      J2I2 = I3**(-2.0D0/3)
      J2I3 = -2.0D0*I2/(3.0D0*I3**(5.0D0/3))
      J3I3 = 1./(2.0D0*sqrt(I3))

      I1E = reshape((/2., 2., 2., 0., 0., 0./), shape(I1E))
      I2E = reshape((/2*C22 + 2*C33, 2*C11 + 2*C33, 2*C11 + 2*C22, -2*C23, -2*C13, -2*C12/), shape(I2E))
      I3E = reshape((/2*C22*C33 - 2*C23**2, 2*C11*C33 - 2*C13**2, 2*C11*C22 - 2*C12**2, &
           -2*C11*C23 + 2*C12*C13, 2*C12*C23 - 2*C13*C22, -2*C12*C33 + 2*C13*C23/), shape(I3E))

      J1E = J1I1*I1E + J1I3*I3E
      J2E = J2I2*I2E + J2I3*I3E
      J3E = J3I3*I3E

!       Sv = reshape(J3E, (/6/))
      Sv = A10*reshape(J1E, (/6/)) + A01*reshape(J2E, (/6/)) + kappa*(J3 - 1)*reshape(J3E, (/6/))
      S = reshape((/Sv(1), Sv(6), Sv(5), &
                    Sv(6), Sv(2), Sv(4), &
                    Sv(5), Sv(4), Sv(3)/), shape(S))

      I2EE = reshape((/0.0D0, 4.0D0, 4.0D0,  0.0D0,  0.0D0,  0.0D0, &
                       4.0D0, 0.0D0, 4.0D0,  0.0D0,  0.0D0,  0.0D0, &
                       4.0D0, 4.0D0, 0.0D0,  0.0D0,  0.0D0,  0.0D0, &
                       0.0D0, 0.0D0, 0.0D0, -2.0D0,  0.0D0,  0.0D0, &
                       0.0D0, 0.0D0, 0.0D0,  0.0D0, -2.0D0,  0.0D0, &
                       0.0D0, 0.0D0, 0.0D0,  0.0D0,  0.0D0, -2.0D0/), shape(I2EE))

      I3EE = reshape((/    0.0D0,  4.0D0*C33,  4.0D0*C22, -4.0D0*C23,      0.0D0,      0.0D0, &
                       4.0D0*C33,      0.0D0,  4.0D0*C11,      0.0D0, -4.0D0*C13,      0.0D0, &
                       4.0D0*C22,  4.0D0*C11,      0.0D0,      0.0D0,      0.0D0, -4.0D0*C12, &
                      -4.0D0*C23,      0.0D0,      0.0D0, -2.0D0*C11,  2.0D0*C12,  2.0D0*C13, &
                           0.0D0, -4.0D0*C13,      0.0D0,  2.0D0*C12, -2.0D0*C22,  2.0D0*C23, &
                           0.0D0,      0.0D0, -4.0D0*C12,  2.0D0*C13,  2.0D0*C23, -2.0D0*C33/), shape(I3EE))

!     second derivatives
      J1I1I3 = -1.0D0/(3.0D0*I3**(4.0D0/3))
      J1I3I3 = 4*I1/(9*I3**(7.0D0/3))
      J2I2I3 = -2.0D0/(3*I3**(5.0D0/3))
      J2I3I3 = 10*I2/(9*I3**(8.0D0/3))
      J3I3I3 = -1.0D0/(4*I3**(3.0D0/2))

      J1EE = J1I1I3*(matmul(I1E, transpose(I3E)) + matmul(I3E, transpose(I1E))) &
               + J1I3I3*matmul(I3E, transpose(I3E)) + J1I3*I3EE
      J2EE = J2I2I3*(matmul(I2E, transpose(I3E)) + matmul(I3E, transpose(I2E))) &
                        + J2I3I3*matmul(I3E, transpose(I3E)) + J2I2*I2EE + J2I3*I3EE
      J3EE = J3I3I3*(matmul(I3E, transpose(I3E))) + J3I3*I3EE

      C_SE = A10*J1EE + A01*J2EE + kappa*(matmul(J3E, transpose(J3E))) + kappa*(J3-1)*J3EE

end subroutine mooney_rivlin_S_Sv_and_C


subroutine neo_hookean_S_Sv_and_C(E, S, Sv, C_SE, mu, kappa)
      implicit none
      real(8), intent(in) :: E(3,3), mu, kappa
      real(8), intent(out) :: S(3,3), Sv(6), C_SE(6,6)
      real(8) :: C(3,3), I1E(6,1), I3E(6,1), I3EE(6,6), J1E(6,1), J3E(6,1), EYE(3,3), J1EE(6,6), J3EE(6,6)
      real(8) :: C11, C22, C33, C23, C13, C12, I1, I3, J3, J1I1, J1I3, J3I3, J1I1I3, J1I3I3, J3I3I3

      EYE = reshape((/1, 0, 0, 0, 1, 0, 0, 0, 1/), shape(EYE))
      C = 2*E + EYE

      C11 = C(1,1)
      C22 = C(2,2)
      C33 = C(3,3)
      C23 = C(2,3)
      C13 = C(1,3)
      C12 = C(1,2)
!     invariants and reduced invariants
      I1  = C11 + C22 + C33
      I3  = C11*C22*C33 - C11*C23**2 - C12**2*C33 + 2*C12*C13*C23 - C13**2*C22

      J3  = sqrt(I3)
!     derivatives
      J1I1 = I3**(-1.0/3.0D0)
      J1I3 = -I1/(3*I3**(4.00D0/3.00D0))
      J3I3 = 1./(2.0D0*sqrt(I3))

      I1E = reshape((/2., 2., 2., 0., 0., 0./), shape(I1E))
      I3E = reshape((/2*C22*C33 - 2*C23**2, 2*C11*C33 - 2*C13**2, 2*C11*C22 - 2*C12**2, &
           -2*C11*C23 + 2*C12*C13, 2*C12*C23 - 2*C13*C22, -2*C12*C33 + 2*C13*C23/), shape(I3E))

      J1E = J1I1*I1E + J1I3*I3E
      J3E = J3I3*I3E

      Sv = mu/2.*reshape(J1E, (/6/)) + kappa*(J3 - 1)*reshape(J3E, (/6/))
      S = reshape((/Sv(1), Sv(6), Sv(5), &
                    Sv(6), Sv(2), Sv(4), &
                    Sv(5), Sv(4), Sv(3)/), shape(S))

      I3EE = reshape((/    0.0D0,  4.0D0*C33,  4.0D0*C22, -4.0D0*C23,      0.0D0,      0.0D0, &
                       4.0D0*C33,      0.0D0,  4.0D0*C11,      0.0D0, -4.0D0*C13,      0.0D0, &
                       4.0D0*C22,  4.0D0*C11,      0.0D0,      0.0D0,      0.0D0, -4.0D0*C12, &
                      -4.0D0*C23,      0.0D0,      0.0D0, -2.0D0*C11,  2.0D0*C12,  2.0D0*C13, &
                           0.0D0, -4.0D0*C13,      0.0D0,  2.0D0*C12, -2.0D0*C22,  2.0D0*C23, &
                           0.0D0,      0.0D0, -4.0D0*C12,  2.0D0*C13,  2.0D0*C23, -2.0D0*C33/), shape(I3EE))

!     second derivatives
      J1I1I3 = -1.0D0/(3.0D0*I3**(4.0D0/3))
      J1I3I3 = 4*I1/(9*I3**(7.0D0/3))
      J3I3I3 = -1.0D0/(4*I3**(3.0D0/2))

      J1EE = J1I1I3*(matmul(I1E, transpose(I3E)) + matmul(I3E, transpose(I1E))) &
               + J1I3I3*matmul(I3E, transpose(I3E)) + J1I3*I3EE
      J3EE = J3I3I3*(matmul(I3E, transpose(I3E))) + J3I3*I3EE

      C_SE = mu/2*J1EE + kappa*(matmul(J3E, transpose(J3E))) + kappa*(J3-1)*J3EE

end subroutine neo_hookean_S_Sv_and_C


subroutine neo_hookean_S_Sv_and_C_2D(E, S, Sv, C_SE, mu, kappa)
      implicit none
      real(8), intent(in) :: E(2,2), mu, kappa
      real(8), intent(out) :: S(2,2), Sv(3), C_SE(3,3)
      real(8) :: C(2,2), I1E(3,1), I3E(3,1), I3EE(3,3), J1E(3,1), J3E(3,1), EYE(2,2), J1EE(3,3), J3EE(3,3)
      real(8) :: C11, C22, C33, C12, I1, I3, J3, J1I1, J1I3, J3I3, J1I1I3, J1I3I3, J3I3I3

      EYE = reshape((/1, 0, 0, 1/), shape(EYE))
      C = 2*E + EYE

      C11 = C(1,1)
      C22 = C(2,2)
      C12 = C(1,2)
      C33 = 1

!     invariants and reduced invariants
      I1  = C11 + C22 + C33
      I3  = C11*C22 - C12**2
      J3  = sqrt(I3)

!     derivatives
      J1I1 = I3**(-1.0/3.0D0)
      J1I3 = -I1/(3*I3**(4.00D0/3.00D0))
      J3I3 = 1./(2.0D0*sqrt(I3))

      I1E = reshape((/2., 2., 0./), shape(I1E))
      I3E = reshape((/2*C22*C33, 2*C11*C33, -2*C12*C33/), shape(I3E))

      J1E = J1I1*I1E + J1I3*I3E
      J3E = J3I3*I3E

      Sv = mu/2.*reshape(J1E, (/3/)) + kappa*(J3 - 1)*reshape(J3E, (/3/))
      S = reshape((/Sv(1), Sv(3), &
                    Sv(3), Sv(2) /), shape(S))

      I3EE = reshape((/ 0.0D0, 4.0D0, 0.0D0, &
                        4.0D0, 0.0D0, 0.0D0, &
                        0.0D0, 0.0D0,-2.0D0 /), shape(I3EE))

!     second derivatives
      J1I1I3 = -1.0D0/(3.0D0*I3**(4.0D0/3))
      J1I3I3 = 4*I1/(9*I3**(7.0D0/3))
      J3I3I3 = -1.0D0/(4*I3**(3.0D0/2))

      J1EE = J1I1I3*(matmul(I1E, transpose(I3E)) + matmul(I3E, transpose(I1E))) &
               + J1I3I3*matmul(I3E, transpose(I3E)) + J1I3*I3EE
      J3EE = J3I3I3*(matmul(I3E, transpose(I3E))) + J3I3*I3EE

      C_SE = mu/2*J1EE + kappa*(matmul(J3E, transpose(J3E))) + kappa*(J3-1)*J3EE

end subroutine neo_hookean_S_Sv_and_C_2D


subroutine kirchhoff_S_Sv_and_C(E, S, Sv, C_SE, E_modul, nu)
    implicit none
    real(8), intent(in) :: E(3,3), E_modul, nu
    real(8), intent(out) :: S(3,3), Sv(6), C_SE(6,6)
    real(8) :: lam, mu, Ev(6)

    lam = nu*E_modul / ((1.0D0 + nu) * (1.0D0 - 2*nu))
    mu  = E_modul / (2.0D0*(1 + nu))

    C_SE = reshape((/lam + 2*mu, lam, lam, 0.0D0, 0.0D0, 0.0D0, &
            lam, lam + 2*mu, lam, 0.0D0, 0.0D0, 0.0D0, &
            lam, lam, lam + 2*mu, 0.0D0, 0.0D0, 0.0D0, &
            0.0D0, 0.0D0, 0.0D0, mu, 0.0D0, 0.0D0, &
            0.0D0, 0.0D0, 0.0D0, 0.0D0, mu, 0.0D0, &
            0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, mu/), (/6,6/))

    Ev = (/E(1,1), E(2,2), E(3,3), 2*E(2,3), 2*E(1,3), 2*E(1,2)/)
    Sv = matmul(C_SE, Ev)
    S = reshape((/Sv(1), Sv(6), Sv(5), &
                    Sv(6), Sv(2), Sv(4), &
                    Sv(5), Sv(4), Sv(3)/), shape(S))

end subroutine kirchhoff_S_Sv_and_C


subroutine kirchhoff_S_Sv_and_C_2D(E, S, Sv, C_SE, E_modul, nu, plane_stress)
    implicit none
    logical, intent(in) :: plane_stress
    real(8), intent(in) :: E(2,2), E_modul, nu
    real(8), intent(out) :: S(2,2), Sv(3), C_SE(3,3)
    real(8) :: lam, mu, Ev(3)

    lam = nu*E_modul / ((1.0D0 + nu) * (1.0D0 - 2*nu))
    mu  = E_modul / (2.0D0*(1 + nu))

!     This is slow, maybe one wants to change that in the future
    if (plane_stress) then
        C_SE = E_modul/(1.0D0-nu**2)*reshape((/1.0D0, nu, 0.0D0, &
                                             nu, 1.0D0, 0.0D0, &
                                             0.0D0, 0.0D0, (1-nu)/2.0D0/), shape(C_SE))
    else
        C_SE = reshape((/lam + 2*mu, lam, 0.0D0, &
                         lam, lam + 2*mu, 0.0D0, &
                         0.0D0, 0.0D0, mu/), shape(C_SE))
    end if

    Ev = (/E(1,1), E(2,2), 2*E(1,2)/)
    Sv = matmul(C_SE, Ev)
    S = reshape((/Sv(1), Sv(3), &
                  Sv(3), Sv(2) /), shape(S))

end subroutine kirchhoff_S_Sv_and_C_2D
