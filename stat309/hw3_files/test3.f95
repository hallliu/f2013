program test3

    implicit none
    real, dimension(7,5) :: a
    real :: b(5), c(7)
    integer :: n, m
    character, parameter :: trans = 'n'
    integer :: dima(2)

    do n = 1, 7
        if (n <= 5) then
            b(n) = n
        end if
        inner: do m = 1, 5
            a(n,m) = n - m
        end do inner
    end do

    write (*,*) a(:,3)

    call sgemv (trans, 7, 5, 1.0, a, 7, b, 1, 0, c, 1)

    write (*, '(7f10.0)') a

    dima = ubound(a)
    write (*,*) dima

end program
