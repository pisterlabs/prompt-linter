from sympy import *
init_printing(use_unicode=True)
from sympy.physics.quantum import TensorProduct

# This is so stupid!
def ConjugateMatrix(A):
    Aconj = A.copy()
    for i,el in enumerate(Aconj):
        Aconj[i] = conjugate(el)
    return Aconj

phi = symbols('phi') #symbols('Phi lambda^2')

Faraday = Matrix([[cos(phi),-sin(phi)],[sin(phi),cos(phi)]])

# Stokes parameters
I,Q,U,V = symbols('I Q U V')
# Go from Stokes to coherency
S = Matrix([[1,1,0,0],[0,0,1,1j],[0,0,1,-1j],[1,-1,0,0]])
# Go from coherency to Stokes
Sinv = Matrix.inv(S)

FF = TensorProduct(Faraday,Faraday)

V_S = Matrix([I, Q, U, V])

# Using David's beam approximation
Ax,Ay = symbols('A_x A_y')
Adiag = Matrix([[Ax,0],[0,Ay]])
AAdiag = TensorProduct(Adiag,Adiag)

Sinv * AAdiag * FF * S * V_S
