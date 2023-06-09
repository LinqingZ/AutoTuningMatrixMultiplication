from client.client import GemmDescriptor, GemmOpInstance, ProfilingResult

class StatefulOptimizer(object):
	def __init__(self):
		self.profile_database = {}
		
	def get_best_gemm_impl(self, gemm_descriptor: GemmDescriptor):
		results = self.profile_database(gemm_descriptor, None)
		if results is None:
			return None
			
		sorted_results = sorted(results, key=lambda t: t[1])
		return sorted_results[0]
		
	def persist_profile_results(self, gemm_descriptor: GemmDescriptor, profiling_result: ProfilingResult):
		self.profile_database[gemm_descriptor] = profiling_result
    # get the profiling result
    # receive request from client, find the best algorithm
    # determine if the algorithm is good or not, return the gemmOpInstance 
    # will be either one result or all run of result

    def get_optimal_gemm_algorithm(gemm_descriptor: GemmDescriptor) -> GemmOpInstance:
        pass
    # return the fast algo after run all?
