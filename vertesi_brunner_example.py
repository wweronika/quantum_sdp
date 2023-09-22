from quantum_sdp_general import *
from see_saw_sdp_flexible_n_outcomes import optimise_p_NL
# from see_saw_sdp_test import get_chsh_coefficients

n_settings_A = 3
n_settings_B = 2
n_outcomes_A = [2, 2, 2]
n_outcomes_B = [3, 2]
d = 3
n_iterations = 10

A = [[0, 0], [0, 0], [-1, 0]]
B = [[0, 0, 0], [-2, 0]]
AB = np.zeros(shape=(n_settings_A, n_settings_B, max(n_outcomes_A), max(n_outcomes_B)))
c = np.full(shape=(n_settings_A, n_settings_B, max(n_outcomes_A), max(n_outcomes_B)), fill_value=-np.inf)

AB[0][0][0][1] = -1
AB[1][0][0][0] = -1
AB[2][0][0][0] = 1
AB[2][0][0][1] = 1
AB[0][1][0][0] = 1
AB[1][1][0][0] = 1
AB[2][1][0][0] = 1

for x, A_x in enumerate(A):
    for a, p_a_given_x in enumerate(A_x):
        y = 0
        for b in range(n_outcomes_B[y]):
            if c[x][y][a][b] == -np.inf:
                c[x][y][a][b] = 0
            c[x][y][a][b] = p_a_given_x
for y, B_y in enumerate(B):
    for b, p_b_given_y in enumerate(B_y):
        x = 0
        for a in range(n_outcomes_A[x]):
            if c[x][y][a][b] == -np.inf:
                c[x][y][a][b] = 0
            c[x][y][a][b] = p_b_given_y

for x in range(n_settings_A):
    for y in range(n_settings_B):
        for a in range(n_outcomes_A[x]):
            for b in range(n_outcomes_B[y]):
                if c[x][y][a][b] == -np.inf:
                    c[x][y][a][b] = 0
                c[x][y][a][b] += AB[x][y][a][b]


bell_coefficients = c

# Test of the flexible optimise_p_NL function with CHSH for safety
# n_settings_A = 2
# n_settings_B = 2
# n_outcomes_A = [2, 2]
# n_outcomes_B = [2, 2]
# d = 2
# n_iterations = 10
# bell_coefficients = get_chsh_coefficients()

rho, A_operators, B_operators, value = optimise_p_NL(
                                        n_settings_A, 
                                        n_settings_B, 
                                        n_outcomes_A, 
                                        n_outcomes_B, 
                                        d, 
                                        bell_coefficients, 
                                        n_iterations
                                        )

print(value)
