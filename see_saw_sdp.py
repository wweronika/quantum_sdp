import cvxpy as cp
import numpy as np
from scipy.stats import unitary_group

n_outcomes = 2
n_settings = 2
d = 2
n_iterations = 10
bell_coefficients = [[[[]]]]
I_A = np.identity(d)
I_B = np.identity(d)

def generate_random_projective_measurement(d, rank):
    if rank > d:
        return None
    random_unitary = unitary_group.rvs(d)
    measurement = np.zeros(shape=(d,d)).astype('complex128')
    for i in range(rank):
        random_ket = np.array([random_unitary[:, i]]).T
        measurement_rank_1 = random_ket @ random_ket.conj().T
        measurement += measurement_rank_1
    return measurement

def sum_of_traces_with_bell_coefficients(A, B, rho, bell_coefficients):
    n_settings_A = len(A) # range of x
    d_A = len(A[0]) # range of a
    n_settings_B = len(B) # range of y
    d_B = len(B[0]) # range of b

    result_of_measurements = 0
    for x in range(n_settings_A):
        for y in range(n_settings_B):
            for a in range(d_A):
                for b in range(d_B):
                    result_of_measurements = result_of_measurements + bell_coefficients[x][y][a][b] * cp.trace(cp.matmul(rho, cp.kron(A[x][a], B[y][b])))
    return cp.real(result_of_measurements)

# A[setting][outcome][x][y]
# i.e. A[0][1] is a rank-1 projector corresponding to A^0_1
A_operators = [[generate_random_projective_measurement(d, 1) for i in range(n_outcomes)] for j in range(n_settings)]
B_operators = [[generate_random_projective_measurement(d, 1) for i in range(n_outcomes)] for j in range(n_settings)]
print(A_operators.shape)

for i in range(n_iterations):
    # 1: maximise rho
    rho = cp.Variable((d, d), hermitian=True)
    constraints = [cp.trace(rho) == 1, rho >> 1]
    problem_rho = cp.Problem(
        cp.Maximize(
            sum_of_traces_with_bell_coefficients(A_operators, B_operators, rho, bell_coefficients)
        ),
        constraints
    )
    problem_rho.solve()
    rho = rho.value
    # 2: maximise {A^x_a}
    A_operators = ([[cp.Variable((d, d), hermitian=True) for i in range(n_outcomes)] for j in range(n_settings)])
    constraints = [cp.sum(A_operators[i]) == I_A for i in range(n_settings)]
    for i in range(n_settings):
        for j in range(n_outcomes):
            constraints.append(A_operators[i][j] >> 0)
    problem_A = cp.Problem(
        cp.Maximize(
            sum_of_traces_with_bell_coefficients(A_operators, B_operators, rho, bell_coefficients)
        ),
        constraints
    )
    problem_A.solve()
    A_operators = [[A_operators[i][j].value for i in range(n_outcomes)] for j in range(n_settings)]
    # 3: maximise {B^y_b}
    B_operators = ([[cp.Variable((d, d), hermitian=True) for i in range(n_outcomes)] for j in range(n_settings)])
    constraints = [cp.sum(B_operators[i]) == I_B for i in range(n_settings)]
    for i in range(n_settings):
        for j in range(n_outcomes):
            constraints.append(B_operators[i][j] >> 0)
    problem_B = cp.Problem(
        cp.Maximize(
            sum_of_traces_with_bell_coefficients(A_operators, B_operators, rho, bell_coefficients)
        ),
        constraints
    )
    problem_B.solve()
    B_operators = [[B_operators[i][j].value for i in range(n_outcomes)] for j in range(n_settings)]

print("rho: ")
print(rho)
print("A_operators: ")
print(A_operators)
print("B_operators: ")
print(B_operators)

