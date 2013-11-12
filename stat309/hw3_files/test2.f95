program test2
    implicit none
    real :: a(5,5), b(5)
    integer :: n, m

    do n = 1, 5
        b(n) = n
        inner: do m = 1, 5
            a(n,m) = n + m
        end do inner
    end do

    write (*,'(5f5.2)') b

    write (*,'(5f6.3)') a

end program
