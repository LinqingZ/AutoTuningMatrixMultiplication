import datetime
from torch import Tensor

from client import GemmDescriptor
from internal.gemm_executor import GemmProvider

class Profiler(object):
	def __init__(self, tensor_a, tensor_b, tensor_c, gemm_descriptor, gemm_impl):
		self.tensor_a = tensor_a 
		self.tensor_b = tensor_b
		self.tensor_c = tensor_c
		self.gemm_descriptor = gemm_descriptor
		self.gemm_impl = gemm_impl

	def profile_instances(self, gemm_descriptor: GemmDescriptor, provider: GemmProvider) -> list[tuple[int, float]]: # run the matrix on all algorithms and return a list of result
		result = [] 
		for gemm_impl in provider.get_gemm_implementations(gemm_descriptor):		
			start = datetime.datetime.now()
			# repeat the profiling a few times to get more reliable results
			provider.execute_gemm(self.tensor_a, self.tensor_b, self.tensor_c, gemm_descriptor, gemm_impl)
			end = datetime.datetime.now()
			time_taken = end - start
			result.append((gemm_impl, time_taken))
		print("profiling result", result)	
		return result