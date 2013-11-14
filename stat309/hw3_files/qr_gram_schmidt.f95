module qr_gram_schmidt
    implicit none

    contains
        subroutine gram_qr (A, Q)
            
            double precision, dimension(:, :) :: A
            double precision, dimension(:, :) :: Q

            integer :: op_col, n
    
            ! We will store R inside A and put Q into its place.
    
            do op_col = 1, size(A, 2)
                ! Copy a_k into the appropriate column of Q
                Q(:, op_col) = A(:, op_col)

                ! Fill in the other r_{jk} by taking inner products
                do n = 1, (op_col - 1)
                    ! Q(:, op_col) currently contains a_k
                    A(n, op_col) = dot_product (Q(:, n), Q(:, op_col))
                end do

                ! Subtract the linear combination of r_{ij}a_j.
                do n = 1, (op_col - 1)
                    Q(:, op_col) = Q(:, op_col) - A(n, op_col) * Q(:, n)
                end do
                
                ! Gets r_{kk}
                A(op_col, op_col) = sqrt (dot_product (Q(:, op_col), Q(:, op_col)))
    
                ! Produces the finalized column of Q by dividing by r_{kk}
                Q(:, op_col) = Q(:, op_col) / A(op_col, op_col)
    
                ! Zero out the rest of them
                A(op_col + 1:, op_col) = 0
            end do

            return
        end subroutine gram_qr
end module
