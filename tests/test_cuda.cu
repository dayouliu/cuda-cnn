#include <iostream>

// CUDA kernel
__global__ void vectorAdd(int *a, int *b, int *c, int size) {
    // block idx is block number, blockDim.x is the dimension of each block
    // tid is the index in the array
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // you need this check because the array may not divide evenly
    // ex: 1000 => 256, 256, 256, 256 => has extra threads
    if(tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int testSimpleCuda() {
    std::cout << "Hello, World!" << std::endl;

    int size = 1024;
    int* vecA = new int[size];
    int* vecB = new int[size];
    int* vecC = new int[size];

    for(int i = 0; i < size; ++i) {
        vecA[i] = i;
        vecB[i] = 2 * i;
    }

    int *vecACuda, *vecBCuda, *vecCCuda;
    cudaMalloc(&vecACuda, size * sizeof(int));
    cudaMalloc(&vecBCuda, size * sizeof(int));
    cudaMalloc(&vecCCuda, size * sizeof(int));

    cudaMemcpy(vecACuda, vecA, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(vecBCuda, vecB, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(vecCCuda, vecC, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // blocking call to cuda
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(vecACuda, vecBCuda, vecCCuda, size);

    cudaMemcpy(vecC, vecCCuda, size * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < size; ++i) {
        if(vecC[i] != vecA[i] + vecB[i]) {
            std::cerr << "incorrect value at index " << i << std::endl;
            break;
        }
        std::cout << vecA[i] << " " << vecB[i] << " " << vecC[i] << std::endl;
    }

    delete[] vecA;
    delete[] vecB;
    delete[] vecC;
    cudaFree(vecACuda);
    cudaFree(vecBCuda);
    cudaFree(vecCCuda);

    return 0;
}
