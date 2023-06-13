from enum import Enum
from dataclasses import dataclass
from torch import Tensor
import itertools
import torch
import random
from internal.gemm_executor import GemmProvider
from internal.profiler import Profiler

from internal.stateful_optimizer import StatefulOptimizer


def TransposeOp(Enum):
	NoTranspose = 0
	Transpose = 1

@dataclass
class GemmDescriptor: 
    # the description of general matrix multiplication need to perform calculation
    m: int
    n: int
    k: int
    stride_a: int
    stride_b: int
    stride_c: int
    op_a: TransposeOp
    op_b: TransposeOp

class AutoTunedGemm(object):
    def __init__(self):
        self.optimizer = StatefulOptimizer()
        self.profiler = Profiler()
        self.provider = GemmProvider.HipblasGemmProvider()
        
    def optimized_gemm(self, tensor_a: Tensor, tensor_b: Tensor, tensor_c: Tensor, gemm_descriptor: GemmDescriptor):
        impl = self.optimizer.get_best_gemm_impl(gemm_descriptor) # check if the database has a same matrix size as the input matrix
        if impl is None:
            profile_results = self.profiler.profile_instances(gemm_descriptor, self.provider)
            self.optimizer.persist_profile_results(gemm_descriptor, profile_results) # store profile result into database
            impl = self.optimizer.get_best_gemm_impl(gemm_descriptor) # after run all the matrix algorithm, find the fastest algorithm searching on the database
        
        return self.provider.execute_gemm(tensor_a, tensor_b, tensor_c, gemm_descriptor, impl)

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
    stride_a, stride_b, stride_c, op_a, op_b = None, None, None, 0, 0
    return matrix_A, matrix_B, matrix_C, m, n, k, stride_a, stride_b, stride_c, op_a, op_b

matrix_A, matrix_B, matrix_C, m, n, k, stride_a, stride_b, stride_c, op_a, op_b = generate_matrices()
print("matrix_A", matrix_A)
print("matrix_B", matrix_B)
print("matrix_C", matrix_C)
auto_gem = AutoTunedGemm()
auto_gem.optimized_gemm(matrix_A, matrix_B, matrix_C, GemmDescriptor(m, n, k, stride_a, stride_b, stride_c, op_a, op_b))