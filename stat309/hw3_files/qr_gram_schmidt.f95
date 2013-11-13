module qr_gram_schmidt
    implicit none

    subroutine gram_qr (A, Q)
        double precision, dimension(:, :) :: A, Q
        integer :: dimA(2)
        integer :: op_col, n, m
        double precision :: anorm

        dimA = ubound(A)
        allocate(proj_vect(dimA(1)))

        ! dimA is the shape of A. We will store R inside A and put Q into its
        ! place.

        do op_col = 1, minval(dimA(2))
            ! Copy a_k into the appropriate column of Q
            call dcopy (dimA(1), A(:, op_col), 1, Q(:, op_col), 1)
            ! Subtract the linear combination of r_{ij}a_j.
            do n = 1, (op_col - 1)
                call daxpy (dimA(1), A(n, op_col), A(:, n), 1, Q(:, op_col), 1)
            end do
            
            ! Gets r_{kk}
            A(op_col, op_col) = dnrm2 (dimA(1), Q(:, n), 1)

            ! Produces the finalized column of Q by dividing by r_{kk}
            Q(:, op_col) = Q(:, op_col) / A(op_col, op_col)

            ! Fill in the other r_{jk} by taking inner products
            do n = 1, (op_col - 1)
                A(n, op_col) = ddot (dimA(1), Q(:, n), 1, A(:, op_col), 1)
            end do

            ! Zero out the rest of them
            A(op_col + 1: dimA(1), op_col) = 0
        end do
    end subroutine
end module
