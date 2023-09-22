import cvxpy as cp
import numpy as np
from scipy.stats import unitary_group
from math import sqrt

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
    n_settings_A = len(A) # range of 
    n_settings_B = len(B) # range of y
    result_of_measurements = 0
    
    for x in range(n_settings_A):
        for y in range(n_settings_B):
            n_outcomes_A = len(A[x])
            n_outcomes_B = len(B[y])
            for a in range(n_outcomes_A):
                for b in range(n_outcomes_B):
                    new_term = cp.trace(bell_coefficients[x][y][a][b] * cp.matmul(rho, cp.kron(A[x][a], B[y][b])))
                    result_of_measurements = result_of_measurements + new_term
    return cp.real(result_of_measurements)

def optimise_p_NL(n_settings_A, n_settings_B, n_outcomes_A, n_outcomes_B, d, bell_coefficients, n_iterations=10):
    # A[setting][outcome][x][y]
    # i.e. A[0][1] is a rank-1 projector corresponding to A^0_1
    A_operators = [[generate_random_projective_measurement(d, 1) for a in range(n_outcomes_A[x])] for x in range(n_settings_A)]
    B_operators = [[generate_random_projective_measurement(d, 1) for b in range(n_outcomes_B[y])] for y in range(n_settings_B)]

    I_A = np.identity(d)
    I_B = np.identity(d)

    for i in range(n_iterations):
        # 1: maximise rho
        # print("maximising rho")
        rho = cp.Variable((d*d, d*d), hermitian=True)
        constraints = [cp.trace(rho) == 1, rho >> 0]
        problem_rho = cp.Problem(
            cp.Maximize(
                sum_of_traces_with_bell_coefficients(A_operators, B_operators, rho, bell_coefficients)
            ),
            constraints
        )
        problem_rho.solve()
        rho = rho.value

        # 2: maximise {A^x_a}
        # print("maximising A")
        A_operators = ([[cp.Variable((d, d), hermitian=True) for a in range(n_outcomes_A[x])] for x in range(n_settings_A)])
        constraints = [cp.sum(A_operators[x]) == I_A for x in range(n_settings_A)]
        for x in range(n_settings_A):
            for a in range(n_outcomes_A[x]):
                constraints.append(A_operators[x][a] >> 0)
        problem_A = cp.Problem(
            cp.Maximize(
                sum_of_traces_with_bell_coefficients(A_operators, B_operators, rho, bell_coefficients)
            ),
            constraints
        )
        problem_A.solve()
        A_operators = [[A_operators[x][a].value for a in range(n_outcomes_A[x])] for x in range(n_settings_A)]

        # 3: maximise {B^y_b}
        # print("maximising B")
        B_operators = ([[cp.Variable((d, d), hermitian=True) for b in range(n_outcomes_B[y])] for y in range(n_settings_B)])
        constraints = [cp.sum(B_operators[y]) == I_B for y in range(n_settings_B)]
        for y in range(n_settings_B):
            for b in range(n_outcomes_B[y]):
                constraints.append(B_operators[y][b] >> 0)
        problem_B = cp.Problem(
            cp.Maximize(
                sum_of_traces_with_bell_coefficients(A_operators, B_operators, rho, bell_coefficients)
            ),
            constraints
        )
        problem_B.solve()
        B_operators = [[B_operators[y][b].value for b in range(n_outcomes_B[y])] for y in range(n_settings_B)]
        # print("Value: " + str(problem_B.value))
    return rho, A_operators, B_operators, problem_B.value
