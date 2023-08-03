from quantum__sdp_general import *
import numpy as np
from math import sqrt

ket_0 = np.array([[1, 0]]).T 
ket_1 = np.array([[0, 1]]).T

print(np.kron(ket_0, ket_0))

rho = 1/sqrt(2) * (np.kron(ket_0, ket_0) @ np.kron(ket_0, ket_0).T + np.kron(ket_1, ket_1)  @ np.kron(ket_1, ket_1).T)
rho = rho.reshape((2,2,2,2))

print(rho)
A_0 = np.array([[1, 0], [0, 0]])
A_1 = np.array([[0, 0], [0, 1]])

A_operators = [A_0, A_1]

find_max_correlated_measurements(A_operators, rho)