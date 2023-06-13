from client import GemmDescriptor


class StatefulOptimizer(object):
	def __init__(self):
		self.profile_database = {}
		
	def get_best_gemm_impl(self, gemm_descriptor: GemmDescriptor):
		results = self.profile_database(gemm_descriptor, None)
		if results is None: # if the database is none, then  result is None which lead to run all the algorithm to create a database
			return None
			
		sorted_results = sorted(results, key=lambda t: t[1]) # sort the database and return the fastest algorithm
		return sorted_results[0]
		
	def persist_profile_results(self, gemm_descriptor: GemmDescriptor, profiling_result: ProfilingResult):
		self.profile_database[gemm_descriptor] = profiling_result