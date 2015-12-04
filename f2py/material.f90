module NeoHookean
  real(8) :: mu, kappa
contains
  subroutine S_Sv_and_C(E, S, Sv, C_SE)
      implicit none
      real(8), intent(in) :: E(3,3)
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
             
!       I2EE = reshape((/0.0D0, 4.0D0, 4.0D0,  0.0D0,  0.0D0,  0.0D0, &
!                        4.0D0, 0.0D0, 4.0D0,  0.0D0,  0.0D0,  0.0D0, &
!                        4.0D0, 4.0D0, 0.0D0,  0.0D0,  0.0D0,  0.0D0, &
!                        0.0D0, 0.0D0, 0.0D0, -2.0D0,  0.0D0,  0.0D0, &
!                        0.0D0, 0.0D0, 0.0D0,  0.0D0, -2.0D0,  0.0D0, &
!                        0.0D0, 0.0D0, 0.0D0,  0.0D0,  0.0D0, -2.0D0/), shape(I2EE))
                            
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
             
  end subroutine S_Sv_and_C
end module NeoHookean

