from quantum_sdp_general import *
from see_saw_sdp import optimise_p_NL, get_chsh_coefficients

A = [[0, 0, 0], [-3, -3, 0]]
B = [[-3, -3, 0], [0, 0, 0]]
AB = [[
    [[3, 0, 0], [3, 3, 0], [0, 0, 0]], # x=0; y=0; a=0,1,2; b=0,1,2
    [[-3, 0, 0], [-3, -3, 0], [0, 0, 0]]  # x=0; y=1; a=0,1,2; b=0,1,2
    ],
    [
    [[3, 3, 0], [0, 3, 0], [0, 0, 0]], # x=1; y=0; a=0,1,2; b=0,1,2
    [[3, 0, 0], [3, 3, 0], [0, 0, 0]] # x=1; y=1; a=0,1,2; b=0,1,2
    ]]

bell_coefficients = coefficients_from_collin_gissin(A, B, AB)
print(bell_coefficients)
rho, A_operators, B_operators = optimise_p_NL(n_settings=2, n_outcomes=3, d=3, bell_coefficients=bell_coefficients)