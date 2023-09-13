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
    p = np.zeros(shape=(n_measurements_A, n_measurements_B, n_outcomes_A, n_outcomes_B)) # index order: c[x][y][a][b]
    for x in range(n_measurements_A):
        for y in range(n_measurements_B):
            for a in range(n_outcomes_A):
                for b in range(n_outcomes_B):
                    # p(a,b|x,y) = 1/4*[ 1 + (-1)^a*<A_x> + (-1)^b*<B_y> + (-1)^(a+b)*<A_x B_y> ]
                    p[x][y][a][b] = 1/4 * (1 + pow(-1, a) * A[x] + pow(-1, b) * B[y]  + pow(-1, a+b) * AB[x][y])
    return p

def coefficients_from_expectation_values(A, B, AB):
    n_outcomes_A = 2
    n_outcomes_B = 2
    n_measurements_A = len(A)
    n_measurements_B = len(B)
    c = np.zeros(shape=(n_measurements_A, n_measurements_B, n_outcomes_A, n_outcomes_B)) # index order: c[x][y][a][b]
    for x, c_A_x in enumerate(A):
        for b in range(n_outcomes_B):
            # Arbitrary selection of y due to no-signalling
            y = 0
            # A_x = p_A(0 | x) - p_A(1 | x)
            # c[x][y][0][b] += c_A_x
            # c[x][y][1][b] -= c_A_x
            c[x][y][1][b] += c_A_x
    for y, c_B_y in enumerate(B):
        for a in range(n_outcomes_A):
            # Arbitrary selection of x due to no-signalling
            x = 0
            # B_y = p_B(0 | y) - p_B(1 | y)
            # c[x][y][a][0] += c_B_y
            # c[x][y][a][1] -= c_B_y
            c[x][y][a][1] += c_B_y
    for x, c_prime_ax in enumerate(AB):
        for y, c_prime_abxy in enumerate(c_prime_ax):
            # c[x][y][0][0] += c_prime_abxy
            # c[x][y][0][1] -= c_prime_abxy
            # c[x][y][1][0] -= c_prime_abxy
            c[x][y][1][1] += c_prime_abxy
    return c

def coefficients_from_collin_gissin(A, B, AB):
    n_outcomes_A = len(A[0])
    n_outcomes_B = len(B[0])
    n_measurements_A = len(A)
    n_measurements_B = len(B)
    c = np.zeros(shape=(n_measurements_A, n_measurements_B, n_outcomes_A, n_outcomes_B)) # index order: c[x][y][a][b]
    for x, A_x in enumerate(A):
        for a, p_a_given_x in enumerate(A_x):
            y = 0
            for b in range(n_outcomes_B):
                c[x][y][a][b] += p_a_given_x
    for y, B_y in enumerate(B):
        for b, p_b_given_y in enumerate(B_y):
            x = 0
            for a in range(n_outcomes_A):
                c[x][y][a][b] += p_b_given_y
    for x in range(n_measurements_A):
        for y in range(n_measurements_B):
            for a in range(n_outcomes_A):
                for b in range(n_outcomes_B):
                    # print(f"x={x}, y={y}, a={a}, b={b}")
                    # print(c[x][y][a][b])
                    # print(AB[x][y][a][b])
                    c[x][y][a][b] += AB[x][y][a][b]
    return c



# sum_x,y,a,b c[x][y][a][b] * p(a,b | x,y)
 
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