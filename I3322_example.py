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

bell_coefficients = coefficients_from_expectation_values(A, B, AB)

file = open("results.csv", "a")

for i in range(50):
    print("Iteration: ", i)
    rho, A_operators, B_operators, value = optimise_p_NL(n_settings=3, n_outcomes=2, d=3, bell_coefficients=bell_coefficients)
    file.write(str(value) + '\n') 
