#include "kernel.h"

__device__ int index2(int r, int c, int cdim) {
    return r*cdim + c;
}

__device__ int index3(int l, int r, int c, int rdim, int cdim) {
    return l*rdim*cdim + r*cdim + c;
}

extern "C" __global__ void cuda_conv_relu(
        double *input, double *conv_layer, double *conv_output, int input_dim, int filter_dim, int output_dim) {
    int filter = blockIdx.x;

    int conv_x = threadIdx.x;
    int conv_y = threadIdx.y;

    int start_x = threadIdx.x * filter_dim;
    int start_y = threadIdx.y * filter_dim;

    double dot_prod = 0;
    for (int r = 0; r < filter_dim; ++r) {
        for (int c = 0; c < filter_dim; ++c) {
            dot_prod += input[index2(start_y+r,start_x+c,input_dim)] * conv_layer[index3(filter,r,c,filter_dim,filter_dim)];
        }
    }

    if (dot_prod < 0) {
        dot_prod = 0;
    }

    conv_output[index3(filter,conv_y,conv_x,output_dim,output_dim)] = dot_prod;
}

extern "C" __global__ void cuda_flatten(double *conv_output, double *linear_layer, double *flatten_output,
                                        int conv_dim, int linear_dim, int flatten_dim) {
    int label_i = blockIdx.x;
    int filter_i = blockIdx.y;

    int conv_x = threadIdx.x;
    int conv_y = threadIdx.y;

    int linear_i = filter_i * 400 + conv_y * conv_dim + conv_x;

    flatten_output[index2(label_i, linear_i, flatten_dim)] =
            conv_output[index3(filter_i,conv_y,conv_x,conv_dim, conv_dim)] *
            linear_layer[index2(label_i, linear_i, linear_dim)];
}

extern "C" __global__ void cuda_sum(double *sum_input, double *sum_output, int sum_in_dim, int sum_out_dim, int sum_batch) {
    int label_i = blockIdx.x;
    int sum_i = threadIdx.x;
    int start = sum_i * sum_batch;
    int end = start + sum_batch;

    double result = 0;
    for (int i = start; i < end; ++i) {
        result += sum_input[index2(label_i, i, sum_in_dim)];
    }

    sum_output[index2(label_i, sum_i, sum_out_dim)] = result;
}

extern "C" __global__ void cuda_out(double *sum_output, double *output, int sum_dim) {
    int label_i = threadIdx.x;

    double result = 0;
    for (int i = 0; i < sum_dim; ++i) {
        result += sum_output[index2(label_i, i, sum_dim)];
    }

    output[label_i] = result;
}