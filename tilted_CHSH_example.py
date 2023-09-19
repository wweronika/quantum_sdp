from quantum_sdp_general import *
from see_saw_sdp import optimise_p_NL
from math import pi, sqrt, tan, atan, sin, cos

theta = pi/8
alpha = 2/sqrt(1 + 2 * pow(tan(2*theta), 2))
mu = atan(sin(2 * theta))

A = [alpha, 0]
B = [0, 0]
AB = [[1, 1], [1, -1]]

sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

bell_coefficients = coefficients_from_expectation_values(A, B, AB)
rho, A_operators, B_operators, value = optimise_p_NL(n_settings=2, n_outcomes=2, d=2, bell_coefficients=bell_coefficients)

correct_A_operators = [sigma_z, sigma_x]
correct_B_operators = [cos(mu) * sigma_z + sin(mu) * sigma_x, cos(mu) * sigma_z - sin(mu) * sigma_x]

print(A_operators, correct_A_operators)
# print(A_operators == correct_A_operators)

# print(B_operators, correct_B_operators)
# print(B_operators == correct_B_operators)

