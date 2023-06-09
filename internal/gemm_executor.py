from abc import ABC, abstractmethod
from torch import Tensor
import torch
from algorithms.hipblas_algo import hipblas_algorithm1, hipblas_algorithm2, hipblas_algorithm3
from algorithms.numpy_algo import numpy_algorithm1, numpy_algorithm2, numpy_algorithm3
from client.client import GemmDescriptor
from internal.profiler import Profiler
from internal.stateful_optimizer import StatefulOptimizer


class GemmStatus: # after execution of GEMM what will return?
    def __init__(self) -> None:
        pass

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
        return gemm_impl(tensor_a, tensor_b, tensor_c, gemm_descriptor)

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
        # implement me
        return gemm_impl(tensor_a, tensor_b, tensor_c, gemm_descriptor)
        


class AutotunedGemm(object):
    def __init__(self):
        self.optimizer = StatefulOptimizer()
        self.profiler = Profiler()
        self.provider = GemmProvider.HipblasGemmProvider()
        
    def optimized_gemm(self, tensor_a: Tensor, tensor_b: Tensor, tensor_c: Tensor, gemm_descriptor: GemmDescriptor):
        impl = self.optimizer.get_best_gemm_impl(gemm_descriptor)
        if impl is None:
            profile_results = self.profiler.profile_instances(gemm_descriptor, self.provider)
            self.optimizer.persist_profile_results(gemm_descriptor)
            impl = self.optimizer.get_best_gemm_impl(gemm_descriptor)
        
        return self.provider.execute_gemm(tensor_a, tensor_b, tensor_c, gemm_descriptor, impl)
