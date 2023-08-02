import numpy as np
from math import sqrt

def ocross(a,b):
    return np.tensordot(a,b, axes=0)

q = 1/5
a = sqrt(131/2)
lambdas = [3257/6884, 450/1721, 450/1721, 27/6884]

ket_0 = np.array([[1,0,0]]).T
ket_1 = np.array([[0,1,0]]).T
ket_2 = np.array([[0,0,1]]).T

a_0_ket = -q * ket_0 + sqrt(3) * q * ket_1 + sqrt(1-4*q**2) * ket_2
a_1_ket = 2 * q * ket_0 + sqrt(1-4*q**2) * ket_2
a_2_ket = -q * ket_0 - sqrt(3) * q * ket_1 + sqrt(1-4*q**2) * ket_2

psi_0_ket = 1/sqrt(2) * (ocross(ket_0, ket_0) + ocross(ket_1, ket_1))
psi_1_ket = a/12 * (ocross(ket_0, ket_1) + ocross(ket_1, ket_0)) + 1/60 * ocross(ket_0, ket_2) - 3/10 * ocross(ket_2, ket_1)
psi_2_ket = a/12 * (ocross(ket_0, ket_0) - ocross(ket_1, ket_1)) + 1/60 * ocross(ket_1, ket_2) + 3/10 * ocross(ket_2, ket_0)
psi_3_ket = 1/sqrt(3) * (-ocross(ket_0, ket_1) + ocross(ket_1, ket_0) + ocross(ket_2, ket_2)) 
psis = [psi_0_ket, psi_1_ket, psi_2_ket, psi_3_ket]


print(a_0_ket.T)
print(a_0_ket)

rho = np.zeros(shape=(3,3,3,3))

for i in range(4):
    rho += lambdas[i] * np.dot(psis[i], psis[i].T).reshape(3,3,3,3)

rho = rho.reshape(9,9)
A0_A0_operator = np.dot(a_0_ket, a_0_ket.T)

# print(rho)
# print(rho.shape)
# print(A0_A0_operator)
# print(A0_A0_operator.shape)

# -----------------------------------------------------------
# CONVEX OPTIMISATION  PART BELOW
# -----------------------------------------------------------

# Import packages.
import cvxpy as cp

# Generate a random SDP.
n = 3
np.random.seed(1)
# M_02 = np.random.randn(n, n)
I = np.identity(n)

# Define and solve the CVXPY problem.
# Create a symmetric matrix variable.
M_02 = cp.Variable((n,n), hermitian=True)
# The operator >> denotes matrix inequality.
constraints = [M_02 >> 0, I-M_02 >> 0]

prob = (cp.Problem
            (cp.Minimize(
                cp.trace(rho @ cp.kron(A0_A0_operator, M_02) + cp.trace(rho @ cp.kron(I - A0_A0_operator, I - M_02)))
                ),
            constraints
            )
)   
prob.solve()

# Print result.
print("The optimal value is", prob.value)
print("A solution M_02 is")
print(M_02.value)
