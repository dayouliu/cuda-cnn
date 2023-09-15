# CUDA CNN

A small library for building and running CNN models and calculations in CUDA.

Requires CUDA Toolkit, tested with v11.5.

`src/main.cu` has an example CNN with a convolutional layer and linear layer.

### main.cu Example

The example model layers are as follows:

```c++
    input[100][100]               // represents the input image             
    conv_layer[10][5][5]          // 10 5x5 convolution filters  
    conv_output[10][20][20]       // resulting conv output
    linear_layer[10][4000]        // linear layer (one for each label)
    flatten[10][4000]             // resulting intermediary flattened output
    sum_output1[10][200]          // intermediary sum (to parallelize)
    sum_output2[10][10];          // intermediary sum (to parallelize)
    output[10];                   // resulting output (ex: labels, weights, etc) 
```

`input`, `conv_layer`, `linear_layer` is read from the specified CSV file.

We load our tensors representing each layer into the CUDA device, and specify CUDA 
calculations in the `src/kernel/kernel.cu` file.

### TODO: Hyperfine benchmarking times

