module qr_householder
    implicit none
    
    contains
        recursive subroutine h_qr (A)

            double precision, dimension(:, :) :: A
            double precision :: alpha, uNorm2

            integer :: k

            ! We're operating on the first column of A, called x. Thus
            ! x_1=A(1,1). Calculate alpha=(+/-) ||x||
            if (A(1,1) > 0) then
                alpha = -sqrt (dot_product (A(:, 1), A(:, 1)))
            else
                alpha = sqrt (dot_product (A(:, 1), A(:, 1)))
            end if

            ! Calculate the vector used in the Householder transform and store
            ! it into the tail of the first column of A

            A(2:, 1) = -A(2:, 1) / (alpha  - A(1,1))

            ! Norm of u squared is the norm of the tail plus the first entry
            ! squared, which is 1
            uNorm2 = dot_product (A(2:, 1), A(2:, 1)) + 1

            ! Apply the transform to each column of A sequentially, except the
            ! first which we treat specially.
            A(1,1) = alpha

            do k = 2, size(A, 2)
                call h_multiply (A(2:, 1), A(:, k), uNorm2)
            end do

            ! Recurse down into the sub-block of A if there's anywhere left to
            ! go. Hopefully the compiler takes care of the stack frame for us.
            if (size(A, 2) < 2) then
                return
            end if

            call h_qr (A(2:, 2:))
        end subroutine h_qr
        

        ! This subroutine applies the Q stored in the lower part of A to x and
        ! stores the result in x.
        subroutine apply_Q (A, x)
            double precision, dimension (:, :) :: A
            double precision, dimension (:) :: x
            double precision :: aNorm2
            integer :: k

            ! Loop through the columns of A, extract the proper refl vector from
            ! them, and multiply. Edge case handling for square matrices is
            ! taken care of in h_multiply.

            do k = 1, size (A, 2)
                aNorm2 = dot_product (A(k + 1:, k), A(k + 1:, k))
                call h_multiply (A(k + 1:, k), x(k:), aNorm2)
            end do
        end subroutine apply_Q

        ! This subroutine applies Q^T to x, where Q is derived from the lower
        ! part of A
        subroutine apply_Q_T(A, x)
            double precision, dimension (:, :) :: A
            double precision, dimension (:) :: x
            double precision :: aNorm2
            integer :: k

            ! Loop through the columns of A, extract the proper refl vector from
            ! them, and multiply. Edge case handling for square matrices is
            ! taken care of in h_multiply.

            do k = size(A, 2), 1, -1
                aNorm2 = dot_product (A(k + 1:, k), A(k + 1:, k))
                call h_multiply (A(k + 1:, k), x(k:), aNorm2)
            end do
        end subroutine apply_Q_T

        ! This subroutine extracts Q from its implicit form in A and puts it
        ! into the Q passed in as an argument.
        subroutine extract_Q(A, Q)
            double precision, dimension (:, :) :: A, Q
            integer :: k

            ! Q is assumed to be square with size equal to num rows of A
            ! Fill it in with a std basis vector and apply Q.
            do k = 1, size(A, 1)
                Q(k, k) = 1
                call apply_Q (A, Q(:, k))
            end do
        end subroutine extract_Q

        ! This subroutine calculates the vector H_u*v and stores it back into v. 
        ! u is expected to have one less dimension than v, under the assumption
        ! that its first entry is 1. Takes the norm of u squared in uNorm2
        subroutine h_multiply(u, v, uNorm2)
            double precision, dimension(:) :: u, v
            double precision :: Umult, uNorm2

            ! Edge case: if u is empty, then it's just scalar multiplication by
            ! 1, so we do nothing and return
            if (size(u) == 0) then
                return
            end if

            ! we take the dot product of the tail entries of u, v, then add in
            ! the contribution from u_1v_1=v_1
            Umult = 2 / uNorm2 * (dot_product (u, v(2:)) + v(1))

            v = -Umult * u + v
        end subroutine h_multiply
end module qr_householder
