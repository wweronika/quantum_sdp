from quantum_sdp_general import *
from see_saw_sdp import optimise_p_NL, get_chsh_coefficients

# A_i expectation value coefficients as seen in the paper
A = [0, -1, 0]
# B expectation value coefficients as seen in the paper
B = [-1, -2, 0]
# AB_ij expectation value coefficients as seen in the paper
AB = [[1, 1, -1], [1, 1, 1], [-1, 1, 0]]

chsh_A = [0,0]
chsh_B = [0,0]
chsh_AB = [[1, 1], [1, -1]]

bell_coefficients = coefficients_from_expectation_values(chsh_A, chsh_B, chsh_AB)
print(bell_coefficients)
chsh_coefficients = get_chsh_coefficients()
print(chsh_coefficients)

print(bell_coefficients == chsh_coefficients)

# bell_coefficients = probabilities_from_expectation_values(A, B, AB)
# print(bell_coefficients.shape)
# print(bell_coefficients)
# rho, A_operators, B_operators = optimise_p_NL(n_settings=3, n_outcomes=2, d=2, bell_coefficients=bell_coefficients)

# print("rho: ")
# print(rho)
# print("A_operators: ")
# print(A_operators)
# print("B_operators: ")
# print(B_operators)