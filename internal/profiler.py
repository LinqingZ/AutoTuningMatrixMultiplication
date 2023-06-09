import datetime
from torch import Tensor
from client.client import GemmDescriptor
from internal.gemm_executor import GemmProvider
from internal.stateful_optimizer import StatefulOptimizer


class AutotunedGemm(object):
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