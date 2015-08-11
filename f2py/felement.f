C FILE: FIB3.F
      SUBROUTINE SCATTER_MATRIX(B, C, N, M, O)
C     ES SIEHT SO AUS, ALS OB DER ERSTE WERT DER RÜCKGABEWERT IST, UND DER ZWEITE WERT IST 
C
C     CALCULATE FIRST N FIBONACCI NUMBERS

      INTEGER N
      INTEGER M
      INTEGER O

      DOUBLE PRECISION C(M, N)
      DOUBLE PRECISION B(M*O, N*O)
      
Cf2py intent(in) c
Cf2py intent(in) n
Cf2py intent(in) m
Cf2py intent(out) B
      
C     use the python indexing style: counters start at 0 and end at M-1
C     in FORTRAN the indexing starts at 1. 
      DO I=0,M-1 
         DO J=0,N-1
            DO K=0,O-1
               B(O*I+K+1, O*J+K+1) = C(I+1,J+1)
            END DO
         END DO
      END DO

      END
C END FILE FIB3.F