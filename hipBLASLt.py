import hipblaslt_binding

# handle = hipblaslt_binding.hipblasLtCreate()
# # Use the hipBLASLt handle and other wrapped functions as needed
# # hipblaslt_binding.hipblasLtDestroy(handle)
# # handle = ...  # Obtain the hipblas handle
# trans_a = 1
# trans_b = 2
# datatype_a = 3
# datatype_b = 4
# datatype_c = 4
# datatype_d = 1
# compute_type = 1

# hipblaslt_binding.groupedGemm(handle, trans_a, trans_b, datatype_a, datatype_b, datatype_c, datatype_d, compute_type)
print(hipblaslt_binding.add(2, 4))