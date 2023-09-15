#pragma once

__device__ int index2(int r, int c, int cdim);

__device__ int index3(int l, int r, int c, int rdim, int cdim);

extern "C" __global__ void cuda_conv_relu(
        double *input, double *conv_layer, double *conv_output, int input_dim, int filter_dim, int output_dim);

extern "C" __global__ void cuda_flatten(double *conv_output, double *linear_layer, double *flatten_output,
                                        int conv_dim, int linear_dim, int flatten_dim);

extern "C" __global__ void cuda_sum(double *flatten_output, double *sum_output,
                                    int flatten_dim, int sum_dim, int sum_batch);

extern "C" __global__ void cuda_out(double *sum_output, double *output, int sum_dim);