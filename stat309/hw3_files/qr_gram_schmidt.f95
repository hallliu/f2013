module qr_gram_schmidt
    implicit none

    contains
        ! This only works on square matrices. The plan is to kludge something
        ! together on the python side that appends random crap to A's columns
        ! and discards the piece of R we don't give a shit about later.
        subroutine gram_qr (A, Q)
            
            double precision, dimension(:, :) :: A
            double precision, dimension(:, :) :: Q

            integer :: k, j
    
            ! We will store R inside A and put Q into its place.
    
            do k = 1, size(A, 2)
                ! Copy a_k into the appropriate column of Q
                Q(:, k) = A(:, k)

                ! Fill in the other r_{jk} by taking inner products
                do j = 1, (k - 1)
                    ! Q(:, k) currently contains a_k
                    A(j, k) = dot_product (Q(:, j), Q(:, k))
                end do

                ! Subtract the linear combination of r_{jk}q_j.
                do j = 1, (k - 1)
                    call daxpy(size(A, 1), -A(j, k), Q(:, j), 1, Q(:, k), 1)
!                    Q(:, k) = Q(:, k) - A(j, k) * Q(:, j)
                end do
                
                ! Gets r_{kk}
                A(k, k) = sqrt (dot_product (Q(:, k), Q(:, k)))
    
                ! Produces the finalized column of Q by dividing by r_{kk}
                Q(:, k) = Q(:, k) / A(k, k)
    
                ! Zero out the rest of them
                A(k + 1:, k) = 0
            end do

            return
        end subroutine gram_qr
end module
