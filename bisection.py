from sympy import * 
from sympy.abc import x
import math

def bisection(f, a, b, tolerance):
    """This program finds a single isolated root of function f over the interval [a, b]
     Inputs:  f, function whose root we seek
              a, b, the interval endpoints
              tolerance, the tolerance level
     Outputs: xs, approximation of the root
              fs, the function value at xs
              n, the total number of iterations
    """
    # Initialize variables.
    n = 0
    xnprev = a
    not_converged = True
    
    # Evaluate function at endpoints.
    fa = f.subs(x, a) 
    fb = f.subs(x, b)

    # Check if input is valid.
    if a >= b or fa * fb == 0 or tolerance <= 0:
        xs = math.nan
        fs = math.nan
        n = math.nan
        print('Invalid input')
        return [xs, fs, n]
    
    while (not_converged):
        xn = (a + b)/2
        fxn = f.subs(x, xn)

        if fa * fxn < 0:
            b = xn
            fb = fxn
        else:
            a = xn
            fa = fxn
        n += 1
        not_converged = Abs(xn - xnprev) > tolerance * (1 + Abs(xn)) or math.fabs(fxn) >= tolerance
        xnprev = xn

    fs = fxn
    xs = xn
    return [xs, fs, n]

if __name__ == '__main__':
    # Bisection Test
    x = symbols('x')
    f = cos(x)-x
    a = 0
    b = 1
    tolerance = 1 * 10**-6
    print('Bisection Test')
    [xs, fs, n] = bisection(f, a, b, tolerance)
    print('xs:', xs)
    print('fs:', fs)
    print('n:', n)
    print()
