from enum import Enum
from dataclasses import dataclass
from torch import Tensor

def TransposeOp(Enum):
	NoTranspose = 0
	Transpose = 1

@dataclass # GemmDescriptor is description of each GEMM algorithm in order to collect the algorithm data after it runs
class GemmDescriptor: # GemmDescriptor is a value will be consider to select a algorithm
    m: int # number of rows in the first matrix (often referred to as matrix A)
    n: int # number of columns in the first matrix (the same as the number of rows in the second matrix)
    k: int # number of columns in the second matrix (often referred to as matrix B)
    stride_a: int
    stride_b: int
    stride_c: int
    op_a: TransposeOp
    op_b: TransposeOp

class GemmOpInstance:
    def __init__(self, algorithm_name, result, runtime) -> None:
        self.algorithm_name = algorithm_name
        self.result = result
        self.runtime = runtime
    # what else should be on this GemmOpInstance?
    # what is a multiplication instance?
    # a instance operation
    # # how can I store this and use? what will the operation istance could be use
    # # what should I returning? instance should be a sub class or GEM provider?
    # name: str
    # operation: str
    # # create a instance of a gemm operation and then use it?
    # operat = GemmOperator()

class ProfilingResult: # Profile the output of the matrix, and store the value into database?
    gemm_descriptor: GemmDescriptor
    profiling_results: dict[GemmOpInstance, float] 
    # here how profiling the result?
    # profile result to client only?

    def optimized_hipblas_gemm(self, matrix_b: Tensor, matrix_c: Tensor, gemm_descriptor: GemmDescriptor) -> None:
        # here selecting a good gem? check if GemmDescriptor is matching a descriptor in the database, if yes then use that algorithm
        # calling the statefulOptimizer to get a good gemm for calculation?
    	pass