from quantum_sdp_general import *
from see_saw_sdp import optimise_p_NL

# A_i expectation value coefficients as seen in the paper
A = [0, -1, 0]
# B expectation value coefficients as seen in the paper
B = [-1, -2, 0]
# AB_ij expectation value coefficients as seen in the paper
AB = [[1, 1, -1], [1, 1, 1], [-1, 1, 0]]

bell_coefficients = probabilities_from_expectation_values(A, B, AB)
print(bell_coefficients.shape)
print(bell_coefficients)
rho, A_operators, B_operators = optimise_p_NL(n_settings=3, n_outcomes=2, d=2, bell_coefficients=bell_coefficients)

print("rho: ")
print(rho)
print("A_operators: ")
print(A_operators)
print("B_operators: ")
print(B_operators)