from sympy import *
init_printing(use_unicode=True)
from sympy.physics.quantum import TensorProduct

# This is so stupid!
def ConjugateMatrix(A):
    Aconj = A.copy()
    for i,el in enumerate(Aconj):
        Aconj[i] = conjugate(el)
    return Aconj

# Elements of the widefield antenna matrix, for both identical and
# differing antenna beams
Axa,Axb,Aya,Ayb = symbols('A_xa A_xb A_ya A_yb')
Aixa,Aixb,Aiya,Aiyb = symbols('A_ixa A_ixb A_iya A_iyb')
Ajxa,Ajxb,Ajya,Ajyb = symbols('A_jxa A_jxb A_jya A_jyb')
Ax,Ay = symbols('A_x A_y')
# Coherency vector in stokes basis
I,Q,U,V = symbols('I Q U V')
Ex,Ey = symbols('E_x,E_y')

E = Matrix([Ex,Ey])
Econj = Matrix([conjugate(Ex),conjugate(Ey)])
# Go from Stokes to coherency, drop the usual factor of 1/2
S = Matrix([[1,1,0,0],[0,0,1,1j],[0,0,1,-1j],[1,-1,0,0]])
# Go from coherency to Stokes
Sinv = Matrix.inv(S)

V_S = Matrix([I, Q, U, V])
V_C = TensorProduct(E,Econj)

A = Matrix([[Axa,Axb],[Aya,Ayb]])
Aconj =  ConjugateMatrix(A)
Ai = Matrix([[Aixa,Aixb],[Aiya,Aiyb]])
Aiconj = ConjugateMatrix(Ai)
Aj = Matrix([[Ajxa,Ajxb],[Ajya,Ajyb]])
Ajconj = ConjugateMatrix(Aj)

Adiag = Matrix([[Axa,0],[0,Ayb]])

AA = TensorProduct(A,A)
AAcomplex = TensorProduct(A,Aconj)
AiAj = TensorProduct(Ai,Aj)
AiAjcomplex = TensorProduct(Ai,Ajconj)

AAdiag = TensorProduct(Adiag,Adiag)

theta,phi = symbols('theta phi')

# Wrong-o 
#Dxt = cos(phi)*cos(theta)
#Dxp = -sin(phi)
#Dyt = sin(phi)*cos(theta)
#Dyp = cos(phi)

#D = Matrix([[Dxt,Dxp],[Dyt,Dyp]])
#DD = TensorProduct(D,D)

nom_stokes = simplify(AiAjcomplex*S*V_S)
nom_vis = simplify(Sinv*nom_stokes)
leakage_beams_full = simplify(Sinv*AiAjcomplex*S)
leakage_beams_identical = simplify(Sinv*AAcomplex*S)
leakage_beams_identical_real = simplify(Sinv*AA*S)

print latex(nom_stokes)
print
print(latex(nom_vis))

