import cmath

def innerprod(a,b,weights):
    assert len(a) == len(b)
    assert len(b) == len(weights)

    s = 0
    for i in range(len(b)):
        s += a[i].conjugate() * b[i] * weights[i]

    return s
