import cvxpy as cp
import numpy as np
from math import sqrt
from see_saw_sdp import get_chsh_coefficients, sum_of_traces_with_bell_coefficients

def get_optimal_values():

    rho = 1/2 * np.array([[1, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 1]])
    A_operators = [[np.array([[1, 0],
                            [0, 0]]),
                    np.array([[0, 0],
                            [0, 1]])        
                    ],
                    [1/2 * np.array([[1, 1],
                            [1, 1]]),
                    1/2 * np.array([[1, -1],
                            [-1, 1]])
                    ]]

    B_operators = [[1/2 * np.array([[1 + 1/sqrt(2), 1/sqrt(2)],
                            [1/sqrt(2), 1-1/sqrt(2)]]),
                    1/2 * np.array([[1-1/sqrt(2), -1/sqrt(2)],
                            [-1/sqrt(2), 1+1/sqrt(2)]])        
                    ],
                    [1/2 * np.array([[1+1/sqrt(2), -1/sqrt(2)],
                            [-1/sqrt(2), 1-1/sqrt(2)]]),
                    1/2 * np.array([[1-1/sqrt(2), 1/sqrt(2)],
                            [1/sqrt(2), 1+1/sqrt(2)]])
                    ]]
    return rho, A_operators, B_operators

rho, A_operators, B_operators = get_optimal_values()
chsh_coefficients = get_chsh_coefficients()
print(chsh_coefficients[0][1][0][1])

value = sum_of_traces_with_bell_coefficients(A_operators, B_operators, rho, chsh_coefficients).value
print("value: ")
print(value)

# for x in range(2):
#     for a in range(2):
#         print(A_operators[x][a].shape)