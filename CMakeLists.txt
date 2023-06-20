cmake_minimum_required(VERSION 3.12)
project(hipblaslt_binding)

add_subdirectory(pybind11)

pybind11_add_module(hipblaslt_binding hipblaslt_binding.cpp)
