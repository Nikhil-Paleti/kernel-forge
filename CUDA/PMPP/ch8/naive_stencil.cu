#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib> 
#include <cmath> // For fabs
#include <iomanip> // For std::fixed and std::setprecision

#define c0 1
#define c1 1
#define c2 1
#define c3 1
#define c4 1
#define c5 1
#define c6 1

void generateRandomMatrix(float *matrix, int rows, int cols, int depth){
    for (int i=0; i< rows*cols*depth; i++){
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}


__global__ void _stencil(float *input, float *output, int M, int N, int K){
    int k = blockIdx.z * blockDim.z + threadIdx.z; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    if (
        (i > 0 && i < N-1) &&
        (j > 0 && j < M-1) &&
        (k > 0 && k < K-1)
    ){
        output[k*M*N + j*N + i] = c0 * input[k*M*N + j*N + i] 
        + c1 * input[k*M*N + j*N + i-1] 
        + c2 * input[k*M*N + j*N + i+1] 
        + c3 * input[k*M*N + (j-1)*N + i] 
        + c4 * input[k*M*N + (j+1)*N + i] 
        + c5 * input[(k-1)*M*N + j*N + i] 
        + c6 * input[(k+1)*M*N + j*N + i];
    }else{
        output[k*M*N + j*N + i] = input[k*M*N + j*N + i];
    }
}

void stencil(float *input, float *output, int M, int N, int K){
    size_t size_input = M*N*K* sizeof(float);
    float *input_d, *output_d;
    cudaMalloc((void**)&input_d, size_input);
    cudaMalloc((void**)&output_d, size_input);
    cudaMemcpy(input_d, input, size_input, cudaMemcpyHostToDevice);
    dim3 block(8, 8, 16);
    dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y, (K + block.z - 1)/block.z);
    _stencil<<<grid, block>>>(input_d, output_d, M, N, K);
    cudaMemcpy(output, output_d, size_input, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
}

void stencil_cpu(const float* input, float* output, int M, int N, int K) {
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < M; ++j) {
            for (int i = 0; i < N; ++i) {
                int idx = k * M * N + j * N + i;

                // Skip boundary access to avoid out-of-bounds
                if (i > 0 && i < N - 1 && j > 0 && j < M - 1 && k > 0 && k < K - 1) {
                    output[idx] =
                        c0 * input[idx] +
                        c1 * input[k * M * N + j * N + (i - 1)] +
                        c2 * input[k * M * N + j * N + (i + 1)] +
                        c3 * input[k * M * N + (j - 1) * N + i] +
                        c4 * input[k * M * N + (j + 1) * N + i] +
                        c5 * input[(k - 1) * M * N + j * N + i] +
                        c6 * input[(k + 1) * M * N + j * N + i];
                } else {
                    output[idx] = input[idx];
                }
            }
        }
    }
}

int main(){
    int M = 256;
    int N = 256;
    int K = 256;
    float *input = new float[M*N*K];
    float *output = new float[M*N*K];
    float *output_cpu = new float[M*N*K];

    generateRandomMatrix(input, M, N, K);
    stencil(input, output, M, N, K);
    stencil_cpu(input, output_cpu, M, N, K);

    float epsilon = 1e-5f;
    bool mismatch_found = false;
    for (int i = 0; i < M*N*K; ++i) {
        if (std::fabs(output[i] - output_cpu[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": GPU = "
                    << std::fixed << std::setprecision(8) << output[i]
                    << ", CPU = " << output_cpu[i] << std::endl;
            mismatch_found = true;
            break; 
        }
    }

    if (!mismatch_found) {
        std::cout << "Verification successful! GPU and CPU outputs match within tolerance." << std::endl;
    }
    
    delete[] input;
    delete[] output;
    delete[] output_cpu;
    return 0;
}