from enum import Enum
from dataclasses import dataclass
from torch import Tensor

from internal.profiler import AutotunedGemm

def TransposeOp(Enum):
	NoTranspose = 0
	Transpose = 1

@dataclass
class GemmDescriptor: # the description of general matrix multiplication need to perform calculation
    m: int
    n: int
    k: int
    stride_a: int
    stride_b: int
    stride_c: int
    op_a: TransposeOp
    op_b: TransposeOp

class ProfilingResult:
    def __init__(self):
        gemm_descriptor: GemmDescriptor(main.m, main.n, main.k, main.stride_a, main.stride_b, main.stride_c,main.op_a, main.op_b)
        GemmOpInstance = AutotunedGemm.optimized_gemm(main.matrix_A, main.matrix_B, main.matrix_C, gemm_descriptor)
        self.profiling_results: dict[GemmOpInstance, float]