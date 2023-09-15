#pragma once

#include <fstream>
#include <cassert>
#include <vector>
#include <sstream>
#include "tensor.cuh"

tensor2::tensor2(uint rows, uint cols): rows(rows), cols(cols) {
    assert(rows != 0 && cols != 0);
    data = new double[rows * cols];
}

tensor2::~tensor2() {
    delete[] data;
    cudaFree(data_cuda);
}

void tensor2::to_device() {
    assert(!in_device);
    cudaMalloc(&data_cuda, rows * cols * sizeof(double));
    cudaMemcpy(data_cuda, data, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    in_device = true;
}

double* tensor2::device_ptr() {
    assert(in_device);
    return data_cuda;
}

void tensor2::to_host() {
    assert(in_device);
    cudaMemcpy(data, data_cuda, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
}

double tensor2::get(int r, int c) const {
    return data[r*cols+c];
}

void tensor2::set(int r, int c, double v) {
    data[r*cols+c] = v;
}


tensor3::tensor3(uint layers, uint rows, uint cols) : layers(layers), rows(rows), cols(cols) {
    assert(layers != 0 && rows != 0 && cols != 0);
    data = new double[layers*rows*cols];
}

tensor3::~tensor3() {
    delete[] data;
    cudaFree(data_cuda);
}

void tensor3::to_device() {
    assert(!in_device);
    cudaMalloc(&data_cuda, layers * rows * cols * sizeof(double));
    cudaMemcpy(data_cuda, data, layers * rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    in_device = true;
}

double* tensor3::device_ptr() {
    assert(in_device);
    return data_cuda;
}

void tensor3::to_host() {
    assert(in_device);
    cudaMemcpy(data, data_cuda, layers * rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
}

double tensor3::get(int l, int r, int c) const {
    return data[l*rows*cols + r*cols + c];
}

void tensor3::set(int l, int r, int c, double v) {
    data[l*rows*cols + r*cols + c] = v;
}

std::ostream& operator<<(std::ostream &stream, const tensor2 &t) {
    for(int i = 0; i < t.rows; ++i) {
        for(int j = 0; j < t.cols; ++j) {
            stream << t.get(i, j) << " ";
        }
        stream << "\n";
    }
    return stream;
}

std::ostream& operator<<(std::ostream &stream, const tensor3 &t) {
    for(int l = 0; l < t.layers; ++l) {
        for (int i = 0; i < t.rows; ++i) {
            for (int j = 0; j < t.cols; ++j) {
                stream << t.get(l, i, j) << " ";
            }
            stream << "\n";
        }
        stream << "\n";
    }
    return stream;
}