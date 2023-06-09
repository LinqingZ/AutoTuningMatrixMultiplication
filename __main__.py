import itertools
import torch
import random
# from client.client import ProfilingResult


def generate_matrices():
    m = random.randint(1, 5)
    n = random.randint(1, 5)
    k = random.randint(1, 5)
    matrix_A = torch.empty(m, n)
    matrix_B = torch.empty(n, k)
    matrix_C = torch.empty(m, k)

    for i, j in itertools.product(range(m), range(n)):
        matrix_A[i][j] = random.randint(1, 100) 

    for i, j in itertools.product(range(n), range(k)):
        matrix_B[i][j] = random.randint(1, 100)
    stride_a, stride_b, stride_c, op_a, op_b = 1, 1, 1, 0, 0
    return matrix_A, matrix_B, matrix_C, m, n, k, stride_a, stride_b, stride_c, op_a, op_b

if __name__ == "__main__":
    matrix_A, matrix_B, matrix_C, m, n, k, stride_a, stride_b, stride_c, op_a, op_b = generate_matrices()
    print("matrix_A", matrix_A)
    print("matrix_B", matrix_B)
    print("matrix_C", matrix_C)
    # client_input = ProfilingResult()
    # print(client_input.profiling_results)
    # still need to figure out how run the file and workflow
