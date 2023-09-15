#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <sstream>
#include "../src/kernel/kernel.h"
#include "../src/data/tensor.cuh"

using namespace std;

uint INPUT_DIM = 100;
uint CONV_LAYER_SIZE = 10;
uint CONV_FILTER_DIM = 5; // should be factor of INPUT_DIM
uint CONV_OUT_DIM = INPUT_DIM / CONV_FILTER_DIM;
uint OUTPUT_DIM = 10;
uint LINEAR_LAYER_SIZE = OUTPUT_DIM;
uint LINEAR_DIM = CONV_OUT_DIM * CONV_OUT_DIM * CONV_LAYER_SIZE;

string IN_FILE_PATH = "../input/simple.csv";
string CNN_FILE_PATH = "../input/cnn.csv";

void read_input(tensor2 &input) {
    ifstream inputFile(IN_FILE_PATH);
    assert(inputFile.is_open());

    string line;
    getline(inputFile, line); // for first line

    for(int i = 0; i < INPUT_DIM; ++i) {
        getline(inputFile, line);
        istringstream is{line};
        string token;

        for(int j = 0; j < INPUT_DIM; ++j) {
            getline(is, token, ',');
            assert(!token.empty());
            input.set(i, j, stod(token));
        }
    }

    inputFile.close();
}

void read_cnn(tensor3 &conv_layer, tensor2 &linear_layer) {
    ifstream cnnFile(CNN_FILE_PATH);
    assert(cnnFile.is_open());

    // Read filter weights
    string line;
    for(int i = 0; i < CONV_LAYER_SIZE; ++i) {
        getline(cnnFile, line);
        istringstream is{line};
        string token;

        for(int j = 0; j < CONV_FILTER_DIM; ++j) {
            for(int k = 0; k < CONV_FILTER_DIM; ++k) {
                getline(is, token, ',');
                assert(!token.empty());
                conv_layer.set(i, j, k, stod(token));
            }
        }
    }

    getline(cnnFile, line); // blank line

    // Read output layer weights
    for(int i = 0; i < LINEAR_LAYER_SIZE; ++i) {
        getline(cnnFile, line);
        istringstream is{line};
        string token;

        for(int j = 0; j < LINEAR_DIM; ++j) {
            getline(is, token, ',');
            assert(!token.empty());
            linear_layer.set(i, j, stod(token));
        }
    }

    cnnFile.close();
}

int main() {
    // Read input file
    tensor2 input{INPUT_DIM, INPUT_DIM};
    read_input(input);

    // Read CNN file
    tensor3 conv_layer(CONV_LAYER_SIZE, CONV_FILTER_DIM, CONV_FILTER_DIM);
    tensor2 linear_layer(LINEAR_LAYER_SIZE, LINEAR_DIM);
    read_cnn(conv_layer, linear_layer);

    tensor3 conv_output(CONV_LAYER_SIZE, CONV_OUT_DIM, CONV_OUT_DIM);
    tensor2 flatten(LINEAR_LAYER_SIZE, LINEAR_DIM);
    tensor2 sum_output1(LINEAR_LAYER_SIZE, 200);
    tensor2 sum_output2(LINEAR_LAYER_SIZE, 10);
    tensor2 output(1, OUTPUT_DIM);

    input.to_device();
    conv_layer.to_device();
    conv_output.to_device();
    linear_layer.to_device();
    flatten.to_device();
    sum_output1.to_device();
    sum_output2.to_device();
    output.to_device();

    cuda_conv_relu<<<10, dim3{20, 20, 1}>>>(
            input.device_ptr(), conv_layer.device_ptr(), conv_output.device_ptr(),
            INPUT_DIM, CONV_FILTER_DIM, CONV_OUT_DIM);

//    conv_output.to_host();
//    cout << conv_output << endl;

    cuda_flatten<<<dim3{10, 10}, dim3{20, 20}>>>(
            conv_output.device_ptr(), linear_layer.device_ptr(), flatten.device_ptr(),
            CONV_OUT_DIM, LINEAR_DIM, LINEAR_DIM);

//    flatten.to_host();
//    cout << flatten << endl;

    cuda_sum<<<10, 200>>>(flatten.device_ptr(), sum_output1.device_ptr(), LINEAR_DIM, 200, 20);

    cuda_sum<<<10, 10>>>(sum_output1.device_ptr(), sum_output2.device_ptr(), 200, 10, 20);

//    sum_output2.to_host();
//    cout << sum_output2 << endl;

    cuda_out<<<1, 10, 0>>>(sum_output2.device_ptr(), output.device_ptr(), 10);

    output.to_host();

    cout << output << endl;

    return 0;
}
