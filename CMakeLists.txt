cmake_minimum_required(VERSION 3.24)
project(cuda_cnn CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(cuda_cnn src/main.cu
        test/test_cuda.cu
        src/kernel/kernel.h
        src/kernel/kernel.cu
        src/data/tensor.cu
        src/data/tensor.cuh
)

set_target_properties(cuda_cnn PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
