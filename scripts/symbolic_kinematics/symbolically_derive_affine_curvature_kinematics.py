import sympy
from sympy import *

L0 = 2

if __name__ == '__main__':
    # Define symbols
    k0, k1, phi, dL = sympy.symbols('k0 k1 phi dL')
    q = sympy.Array([k0, k1, phi, dL])
    s = sympy.symbols('s')
    s_prime = sympy.symbols('s_prime')
    L0 = sympy.symbols('L0')

    # curvature
    kappa = k0 + k1 * s
    # orientation
    alpha = k0 * s + k1 * s ** 2 / 2

    R_phi = sympy.Matrix([
        [sympy.cos(phi), -sympy.sin(phi), 0],
        [sympy.sin(phi), sympy.cos(phi), 0],
        [0, 0, 1],
    ])
    R_alpha = sympy.Matrix([
        [1, 0, 0],
        [0, sympy.cos(alpha), -sympy.sin(alpha)],
        [0, sympy.sin(alpha), sympy.cos(alpha)],
    ])

    # combine rotations
    R = R_phi * R_alpha * R_phi.T
    print("R")
    print(R[0, :])
    print(R[1, :])
    print(R[2, :])

    # integrate trigonometric functions for translational components
    alpha_prime = alpha.subs(s, s_prime)
    int_cos = sympy.integrate(sympy.cos(alpha_prime), (s_prime, 0, s))
    int_sin = sympy.integrate(sympy.sin(alpha_prime), (s_prime, 0, s))

    t = (L0 + dL) * sympy.Matrix([
        [int_sin * sympy.sin(phi)],
        [- int_sin * sympy.cos(phi)],
        [int_cos],
    ])
    print("t")
    print(t[0, 0])
    print(t[1, 0])
    print(t[2, 0])

    # transformation matrix
    T = sympy.BlockMatrix([
        [R, t],
        [sympy.zeros(1, 3), sympy.ones(1, 1)],
    ])
