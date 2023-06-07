from abc import ABC, abstractmethod
import datetime
from torch import Tensor
from client.client import GemmDescriptor
from typing import Dict, List, Tuple

from internal.profiler import Profiler
from internal.stateful_optimizer import StatefulOptimizer


class GemmProvider(ABC):
	def __init__(self):
		pass
		
	@abstractmethod
	def get_gemm_implementations(self, gemm_descriptor: GemmDescriptor) -> List[int]:
		pass
	
	@abstractmethod
	def execute_gemm(self, tensor_a: Tensor, tensor_b: Tensor, tensor_c: Tensor, gemm_descriptor: GemmDescriptor, gemm_impl: int) -> GemmStatus:
		pass
	
class NumPyGemmProvider(GemmProvider):
    def __init(self):
        pass
	
 	def get_gemm_implementations(self, gemm_descriptor: GemmDescriptor) -> List[int]:
     	pass
		# implement me
	
	def execute_gemm(self, tensor_a: Tensor, tensor_b: Tensor, tensor_c: Tensor,  gemm_descriptor: GemmDescriptor, gemm_impl: int) -> GemmStatus:
		# implement me
		pass

class HipblasGemmProvider(GemmProvider):
	def __init(self):
		pass
		
	def get_gemm_implementations(self, gemm_descriptor: GemmDescriptor) -> List[int]:
		# implement me
	
	def execute_gemm(self, tensor_a: Tensor, tensor_b: Tensor, tensor_c: Tensor,  gemm_descriptor: GemmDescriptor, gemm_impl: int) -> GemmStatus:
		# implement me
		pass
		


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
