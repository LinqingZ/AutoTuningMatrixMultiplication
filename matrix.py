# [x] Look for all the numpy matrix algorithms
# [x] Setup basic python code and files to run each matrix
# [x] Set time of each matrix running
# [x] comparison of calculation time
# [ ] Selection of a matrix algorithm
    # [ ] figure out way to select the best matrix to solve, and what, next time use the same one as the size is the same?
    # [ ] following up questions: when running a algorithm 1000 times to know if that algorithm is fast? when solving 
    # running the same matrix for 1000 times to see if that algorithm is the fastest, it may affect of the run time it use

# Questions: 
    # the run time of each algorithm is various, how to solve this issues?

import time
import timeit
import numpy as np

def generate_random_matrices(rows, columns):
    matrix1 = np.random.randint(100, size=(rows, columns))
    matrix2 = np.random.randint(100, size=(rows, columns))
    return matrix1, matrix2
matrix_A, matrix_B = generate_random_matrices(10, 10)

def algorithm1(x, y):
    x = x.numpy()
    y = y.numpy()
    return np.dot(x, y)

def algorithm2(x, y):
    x = x.numpy()
    y = y.numpy()
    return np.matmul(x, y)

def algorithm3(x, y):
    if isinstance(x, np.matrix) and isinstance(y, np.matrix):
        return x*y
    try:
        x = np.mat(x.numpy())
        y = np.mat(y.numpy())
        return x*y
    except Exception:
        return None

# check time of running one time
def time_function(func, *args):
    start_time = time.time()
    func(*args)
    return time.time() - start_time

# time1 = time_function(algorithm1, matrix_A, matrix_B)
# print(f"Algorithm 1 took {time1} seconds")

# time2 = time_function(algorithm2, matrix_A, matrix_B)
# print(f"Algorithm 2 took {time2} seconds")


# time3 = time_function(algorithm3, matrix_A, matrix_B)
# print(f"Algorithm 2 took {time3} seconds")


# check time of running 1000 times
def time_thousand_function(func, *args):
    return timeit.timeit(lambda:func(*args), number=1000)
# print("Running time 1")
# print("thousand times of algorithm 1",time_thousand_function(algorithm1, matrix_A, matrix_B))
# print("thousand times of algorithm 2",time_thousand_function(algorithm2, matrix_A, matrix_B))
# print("thousand times of algorithm 3",time_thousand_function(algorithm3, matrix_A, matrix_B))

# print("Running time 2")
# print("thousand times of algorithm 1", timeit.timeit(lambda:algorithm1(matrix_A, matrix_B), number=1000))
# print("thousand times of algorithm 2", timeit.timeit(lambda:algorithm2(matrix_A, matrix_B), number=1000))
# print("thousand times of algorithm 3", timeit.timeit(lambda:algorithm3(matrix_A, matrix_B), number=1000))

# print("Running time 3")
# print("thousand times of algorithm 1", timeit.timeit(lambda:algorithm1(matrix_A, matrix_B), number=1000))
# print("thousand times of algorithm 2", timeit.timeit(lambda:algorithm2(matrix_A, matrix_B), number=1000))
# print("thousand times of algorithm 3", timeit.timeit(lambda:algorithm3(matrix_A, matrix_B), number=1000))



# framework to do the calculation
# task, algorithm on a problem, build a software, dimension of matrix, figure which algo is the best for that prob
# then, run experiment, next time see the similar then choose a algo,
# two interface, client, give matrix problems, a b c matrix, need row and columns of matrix, stride of matrix, pointer in c, 
# stride as a input in interface, 
# subsystem - run the problem to all the algorithm and give result of all the algorithm output
# profiling system - determine which algorithm is best suited for a particular task or dataset, measure the performance of 
    # different matrix algorithms. By instrumenting the code of the algorithms and measuring their runtime, memory usage, and 
    # other performance metrics, you can compare the efficiency of different matrix algorithms and identify areas for optimization.
    # optimal stride on matrix multiplication problem
# mechanism system - store and run quick in next time


# data you need to pass into hipBLAS and customer
# python method signature, what data and expectation need in the system, interaction the system below it
# performance the computation

# save the results of the profiling the into 


# either way?
# think about the design, think about the implement a python interface with hipBLAS
# data structure of store of store the file...