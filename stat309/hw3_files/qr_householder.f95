module qr_householder
    implicit none
    
    double precision, external :: ddot, dnrm2

    contains
        recursive subroutine h_qr (A)

            double precision, dimension(:, :) A

            integer :: dimA(2)
            integer :: k, i, j

            dimA = ubound(A)

            do k = 1, minval(dimA)
        

        subroutine h_multiply(u, v)
            ! This subroutine multiplies H_u by v and stores the result in v
            double precision, dimension(:) :: u, v
            integer :: vLen
            double precision :: normU2, Umult

            vLen = size(u)

            normU2 = dnrm2 (vLen, u, 1)
            normU2 = normU2 * normU2

            Umult = 2 / normU2 * ddot (vLen, u, 1, v, 1)

            call daxpy (vLen, -Umult, u, 1, v, 1)
        end subroutine h_multiply

