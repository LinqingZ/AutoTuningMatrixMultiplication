from abc import ABC, abstractmethod
from torch import Tensor
from algorithms.hipblas_algo import hipblas_algorithm1, hipblas_algorithm2, hipblas_algorithm3
from algorithms.numpy_algo import numpy_algorithm1, numpy_algorithm2, numpy_algorithm3
from client import GemmDescriptor


class GemmStatus: # return if the calculation is success or not
    def __init__(self, success, calculate_result = None):
        self.success = success
        self.calculate_result = calculate_result
        if self.success:
            print("Calculation was successful!")
            print("Result:", self.calculate_result)
        else:
            print("Calculation failed.")

class GemmProvider(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_gemm_implementations(self, gemm_descriptor: GemmDescriptor) -> list[int]:
        pass

    @abstractmethod
    def execute_gemm(self, tensor_a: Tensor, tensor_b: Tensor, tensor_c: Tensor, gemm_descriptor: GemmDescriptor, gemm_impl: int) -> GemmStatus:
        pass

class NumPyGemmProvider(GemmProvider):
    def __init__(self):
        pass

    def get_gemm_implementations(self, gemm_descriptor: GemmDescriptor) -> list[int]: # what is expecting in the getting all the gemm implementation?
        numpy_algorithm =[
            (numpy_algorithm1, gemm_descriptor),
            (numpy_algorithm2, gemm_descriptor),
            (numpy_algorithm3, gemm_descriptor)
        ]
        return numpy_algorithm

    def execute_gemm(self, tensor_a: Tensor, tensor_b: Tensor, tensor_c: Tensor,  gemm_descriptor: GemmDescriptor, gemm_impl: int) -> GemmStatus:
        try:
            result = gemm_impl(tensor_a, tensor_b, tensor_c, gemm_descriptor)
            return GemmStatus(success=True, calculate_result = result)
        except Exception:
            return GemmStatus(success=False)

class HipblasGemmProvider(GemmProvider):
    def __init__(self):
        pass
        
    def get_gemm_implementations(self, gemm_descriptor: GemmDescriptor) -> list[int]:
        hipblas_algorithm =[
            (hipblas_algorithm1, gemm_descriptor),
            (hipblas_algorithm2, gemm_descriptor),
            (hipblas_algorithm3, gemm_descriptor)
        ]
        return hipblas_algorithm
    
    def execute_gemm(self, tensor_a: Tensor, tensor_b: Tensor, tensor_c: Tensor,  gemm_descriptor: GemmDescriptor, gemm_impl: int) -> GemmStatus:
        try:
            result = gemm_impl(tensor_a, tensor_b, tensor_c, gemm_descriptor)
            return GemmStatus(success=True, calculate_result = result)
        except Exception:
            return GemmStatus(success=False)
        


