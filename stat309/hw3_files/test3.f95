program test3

    implicit none
    real :: a(5,5), b(5), c(5)
    integer :: n, m
    character, parameter :: trans = 'n'

    do n = 1, 5
        b(n) = n
        inner: do m = 1, 5
            a(n,m) = n + m
        end do inner
    end do

    write (*,'(5f5.2)') b

    write (*,'(5f6.3)') a

    call sgemv (trans, 5, 5, 1.0, a, 5, b, 1, 0, c, 1)

    write (*, '(5f10.0)') c
    write (*,*) c(5)

end program
