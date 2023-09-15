#pragma once

#include <iostream>

// InputMatrix needs to use double arrays for CUDA
// we cannot use vectors

class tensor2 {
public:
    uint rows, cols;

    tensor2(uint rows, uint cols);

    ~tensor2();

    void to_device();

    double* device_ptr();

    void to_host();

    double get(int r, int c) const;
    void set(int r, int c, double v);

    // TODO: copy and move assignment operators
    tensor2(const tensor2 &other) = delete;
    tensor2(tensor2 &&other) = delete;
    tensor2& operator=(const tensor2& other) = delete;
    tensor2& operator=(tensor2&& other) = delete;

private:
    double *data;
    double *data_cuda;
    bool in_device = false;
};

struct tensor3 {
    uint layers, rows, cols;

    tensor3(uint layers, uint rows, uint cols);

    ~tensor3();

    void to_device();

    double* device_ptr();

    void to_host();

    double get(int l, int r, int c) const;

    void set(int l, int r, int c, double v);

    // TODO: copy and move assignment operators
    tensor3(const tensor3 &other) = delete;
    tensor3(tensor3 &&other) = delete;
    tensor3& operator=(const tensor3& other) = delete;
    tensor3& operator=(tensor3&& other) = delete;

private:
    double *data;
    double *data_cuda;
    bool in_device = false;
};

std::ostream& operator<<(std::ostream &stream, const tensor2 &t);

std::ostream& operator<<(std::ostream &stream, const tensor3 &t);