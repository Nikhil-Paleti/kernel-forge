/**
 * @file privatization_histogram.cu
 * @brief GPU and CPU histogram comparison example using CUDA.
 *
 * computes histogram using naive atomic operations and privatization in cuda. 
 *
 * Author: Nikhil Paleti
 * Date: June 1, 2025
 */
 
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib> // for rand()
#include <cstring> // for strlen
#include <cctype>  // for std::tolower
#include <cmath> // For fabs
#include <iomanip> // For std::fixed and std::setprecision

#define BIN_SIZE 4
#define NUM_BINS  ((26 + BIN_SIZE - 1) / BIN_SIZE)
#define CFACTOR 2

void cpu_histogram(char *data, unsigned int *hist, unsigned int length) {
    for (unsigned int i = 0; i < length; i++) {
        char c = std::tolower(data[i]);
        int alphabet_position = c - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            hist[alphabet_position / BIN_SIZE]++;
        }
    }
}

__global__ void histo_kernel(char *data, unsigned int length, unsigned int *histo){
    __shared__ unsigned int histo_s[NUM_BINS];
    for(int i = threadIdx.x; i<NUM_BINS; i+= blockDim.x){
        histo_s[i] = 0;
    }
    __syncthreads();

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i=x; i< length; i+=blockDim.x*gridDim.x){
        char c = data[i];
        int alphabet_position = c - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo_s[alphabet_position / BIN_SIZE]), 1);
        }
    }
    __syncthreads();
        
    for(int i = threadIdx.x; i<NUM_BINS; i+= blockDim.x){
        unsigned int binValue = histo_s[i];
        if (binValue > 0){
            atomicAdd(&(histo[i]), binValue);
        }
    }

    
}

void histogram(char *data, unsigned int length, unsigned int *histo){
    dim3 block(4, 1, 1);
    unsigned int ele_block = block.x * CFACTOR;
    dim3 grid((length + ele_block - 1) / ele_block, 1, 1);
    
    size_t size_data = length * sizeof(char);
    size_t size_histo = NUM_BINS * sizeof(unsigned int);
    
    char *data_d;
    unsigned int *histo_d;
    
    cudaMalloc((void**)&data_d, size_data);
    cudaMalloc((void**)&histo_d, size_histo);
    cudaMemcpy(data_d, data, size_data, cudaMemcpyHostToDevice);
    cudaMemset(histo_d, 0, size_histo);

    histo_kernel<<<grid, block>>>(data_d, length, histo_d);
    
    cudaDeviceSynchronize();

    cudaMemcpy(histo, histo_d, size_histo, cudaMemcpyDeviceToHost);
    cudaFree(data_d);
    cudaFree(histo_d);
}

int main() {

    char data[] = "programming massively parallel processors";
    unsigned int length = strlen(data);
    unsigned int hist_CPU[NUM_BINS] = {0};  
    unsigned int hist_GPU[NUM_BINS] = {0};  

    histogram(data, length, hist_GPU);
    cpu_histogram(data, hist_CPU, length);

    float epsilon = 1e-5f;
    bool mismatch_found = false;
    for (int i = 0; i < NUM_BINS; ++i) {
        if (std::fabs(hist_CPU[i] - hist_GPU[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": GPU = "
                    << std::fixed << std::setprecision(8) << hist_GPU[i]
                    << ", CPU = " << hist_CPU[i] << std::endl;
            mismatch_found = true;
            break; 
        }
    }

    if (!mismatch_found) {
        std::cout << "Verification successful! GPU and CPU outputs match within tolerance." << std::endl;
    }

    return 0;
}