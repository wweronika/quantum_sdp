from quantum_sdp_general import *
import numpy as np
from math import sqrt

N = 2
kets = []

for i in range(N):
    ket = np.zeros(shape=(1, N)).T
    ket[i, 0] = 1
    kets.append(ket)

rho = np.zeros(shape=(N, N))
for ket in kets:
    rho += np.kron(ket, ket) @ np.kron(ket, ket).T
rho = rho / N



# print(rho)
# A_0 = np.array([[1, 0], [0, 0]])
# A_1 = np.array([[0, 0], [0, 1]])

# A_operators = [A_0, A_1]

# find_max_correlated_measurements(A_operators, rho)