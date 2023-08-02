import numpy as np
from math import sqrt

def ocross(a,b):
    return np.tensordot(a,b, axes=0)


i_plus_ket = 1/sqrt(2) * np.array([[1, 1j]]).T
i_minus_ket = 1/sqrt(2) * np.array([[1, -1j]]).T

rho_i_plus = i_plus_ket @ np.conj(i_plus_ket.T)
rho_i_minus = i_minus_ket @ np.conj(i_minus_ket.T)

print(rho_i_plus)
print(rho_i_minus)


# -----------------------------------------------------------
# CONVEX OPTIMISATION  PART BELOW
# -----------------------------------------------------------

# Import packages.
import cvxpy as cp

n = 2
I = np.identity(n)

# Define and solve the CVXPY problem.
# Create a Hermitian matrix variable.
M = cp.Variable((n,n), complex=True)
# The operator >> denotes matrix inequality.
constraints = [M >> 0, I-M >> 0]

prob = (cp.Problem
            (cp.Maximize(
                cp.real(cp.trace(rho_i_plus @ M)  + cp.trace(rho_i_minus @ (I - M)))
                ),
            constraints
            )
)
prob.solve()

# Print result.
print("The optimal value is", prob.value)
print("A solution M is")
print(M.value)
