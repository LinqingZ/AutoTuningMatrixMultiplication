import numpy as np

num_matrices = 1000
min_size = 2
max_size = 1000

# Generate 1000 different integer matrices with random sizes
matrices = []
for _ in range(num_matrices):
    rows = np.random.randint(low=min_size, high=max_size+1)
    cols = np.random.randint(low=min_size, high=max_size+1)
    matrix = np.random.randint(low=1, high=10, size=(rows, cols))
    matrices.append(matrix)

# Print the matrices
for i, matrix in enumerate(matrices):
    print(f"Matrix {i+1}:")
    print(matrix)
    print()