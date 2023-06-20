To Do:
- Connect the python python to call the hipblas API the similar methods
https://github.com/ROCmSoftwarePlatform/hipBLASLt/blob/develop/clients/samples/example_hipblaslt_groupedgemm_get_all_algos.cpp
- action item to do this:
    - connect c++ use the pybind libray
    - find all the function need to use in python
        - hipblaslt_ext::getAllAlgos
        - hipblaslt_ext::GroupedGemm groupedGemm
        - std::vector<hipblaslt_ext::GemmInputs>
        - groupedGemm.run
        - hipEventElapsedTime
        - hipblasLtMatmul
    - use ctype to motify the files