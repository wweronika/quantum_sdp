from quantum_sdp_general import *
from see_saw_sdp import optimise_p_NL, get_chsh_coefficients

A = [0,0]
B = [0,0]
AB = [[1, 1], [1, -1]]


bell_coefficients = coefficients_from_expectation_values_1(A, B, AB)
rho, A_operators, B_operators, value = optimise_p_NL(n_settings=2, n_outcomes=2, d=2, bell_coefficients=bell_coefficients)

correct_coefficients = get_chsh_coefficients()
print(bell_coefficients == correct_coefficients)

print(value)
