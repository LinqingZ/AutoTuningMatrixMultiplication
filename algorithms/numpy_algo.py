# Temporary sample algorithms for matrix multiplication  
import torch


def numpy_algorithm1(tensor_a, tensor_b, tensor_c, gemm_descriptor):
    # tensor_c = torch.empty((tensor_a.size(0), tensor_b.size(1)))
    tensor_c = torch.empty((gemm_descriptor.m, gemm_descriptor.k))
    torch.matmul(tensor_a, tensor_b, out=tensor_c)

def numpy_algorithm2(tensor_a, tensor_b, tensor_c, gemm_descriptor):
    tensor_c = torch.empty((gemm_descriptor.m, gemm_descriptor.k))
    torch.einsum('ij,jk->ik', tensor_a, tensor_b, out=tensor_c)

def numpy_algorithm3(tensor_a, tensor_b, tensor_c, gemm_descriptor):
    tensor_c = torch.empty((gemm_descriptor.m, gemm_descriptor.k))
    torch.mm(tensor_a, tensor_b, out=tensor_c)