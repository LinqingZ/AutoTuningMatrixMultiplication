#include <pybind11/pybind11.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace py = pybind11;


void groupedGemmWrapper(py::object handle,
                        bool trans_a,
                        bool trans_b,
                        hipblasDatatype_t datatype_a,
                        hipblasDatatype_t datatype_b,
                        hipblasDatatype_t datatype_c,
                        hipblasDatatype_t datatype_d,
                        hipblasLtComputeType_t compute_type) {
    // Get the hipblas handle from the Python object
    hipblasHandle_t hipblas_handle = py::cast<hipblasHandle_t>(handle);

    // Call the original function
    hipblaslt_ext::GroupedGemm(hipblas_handle,
                               trans_a,
                               trans_b,
                               datatype_a,
                               datatype_b,
                               datatype_c,
                               datatype_d,
                               compute_type);
}


PYBIND11_MODULE(hipblaslt_binding, m) {
    // Wrap hipblaslt functions here using Pybind11
    // m.def("hipblasLtCreate", &hipblasLtCreate, "Create hipBLASLt handle");
    // m.def("hipblasLtDestroy", &hipblasLtDestroy, "Destroy hipBLASLt handle");
    // m.def("get_all_algos", &hipblaslt_ext::getAllAlgos, py::return_value_policy::copy);
    // m.def("groupedGemm", &groupedGemmWrapper, "Perform GroupedGemm using hipblaslt_ext");
    m.def("groupedGemm", &groupedGemmWrapper, "Perform GroupedGemm with pybind11");
}


