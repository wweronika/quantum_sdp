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