import numpy as np
import cvxpy as cp
 
A_operators = [] # list of N x N numpy matrices
rho = [] # M x M numpy array where M = N + dimension of B_i measurement operators

def ocross(a,b):
    return np.tensordot(a,b, axes=0)
 
def sum_of_traces(A, B, rho):
    d = rho.shape[0]
    result_of_measurements =  np.zeros(shape=(d,d))
    for i in range(len(A)):
        result_of_measurements = result_of_measurements + cp.matmul(rho, cp.kron(A[i], B[i]))
    return cp.real(cp.trace(result_of_measurements))

def probabilities_from_expectation_values(A, B, AB):
    # Assume all measurements are qubits, i.e. have 2 outcomes: 0 (value 1) or 1 (value -1)
    n_outcomes_A = 2
    n_outcomes_B = 2
    n_measurements_A = len(A)
    n_measurements_B = len(B)
    c = np.zeros(shape=(n_measurements_A, n_measurements_B, n_outcomes_A, n_outcomes_B)) # index order: c[x][y][a][b]
    for x in range(n_measurements_A):
        for y in range(n_measurements_B):
            for a in range(n_outcomes_A):
                for b in range(n_outcomes_B):
                    c[x][y][a][b] = 1/4 * (1 + pow(-1, a) * A[x] + pow(-1, b) * B[y]  + pow(-1, a+b) * AB[x][y])
 
def find_max_correlated_measurements(A_operators, rho):
    if (len(rho.shape) != 2 
        or rho.shape[0] != rho.shape[1]
        or A_operators[0].shape[0] != A_operators[0].shape[1]) : 
        print("Invalid dimensions of rho and A")
        return
    n_measurements = len(A_operators)
    d_A = A_operators[0].shape[0]
    d_B = rho.shape[0] - d_A
    I_B = np.identity(d_B)
    B_operators = []
    constraints = []

    for i in range(n_measurements):
        B_i = cp.Variable((d_B, d_B), hermitian=True)
        B_operators.append(B_i)
        constraints.append(B_i >> 0)
    constraints.append(cp.sum(B_operators) == I_B)
    problem = (cp.Problem
            (cp.Maximize(
                sum_of_traces(A_operators, B_operators, rho)
                ),
            constraints
            )
    )
    problem.solve()
    print(problem.value)
    print(B_operators)
    for B_i in B_operators:
        print(B_i.value)
    return B_operators