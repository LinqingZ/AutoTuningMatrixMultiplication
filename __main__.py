import itertools
import torch
import random



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
    
    return matrix_A, matrix_B, matrix_C, m, n, k

def main():
    matrix_A, matrix_B, matrix_C, m, n, k = generate_matrices()
    print("matrix_A", matrix_A)
    print("matrix_B", matrix_B)
    print("matrix_C", matrix_C)


if __name__ == "__main__":
    main()
