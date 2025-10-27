#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <iomanip>

#define c0 1
#define c1 1
#define c2 1
#define c3 1
#define c4 1
#define c5 1
#define c6 1

#define ORDER 1
#define INPUT_TILE 8
#define OUTPUT_TILE (INPUT_TILE-2*ORDER)

void generateRandomMatrix(float *matrix, int rows, int cols, int depth) {
    for (int i = 0; i < rows * cols * depth; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

__global__ void _stencil(float *input, float *output, int M, int N, int K) {
    int k = blockIdx.z * OUTPUT_TILE + threadIdx.z - 1;
    int j = blockIdx.y * OUTPUT_TILE + threadIdx.y - 1;
    int i = blockIdx.x * OUTPUT_TILE + threadIdx.x - 1;

    __shared__ float in_s[INPUT_TILE][INPUT_TILE][INPUT_TILE];
    if(
        (i >= 0 && i < N) &&
        (j >=0 && j < M) &&
        (k >=0 && k < K)
    ){
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = input[k*M*N + j*N + i];
    }
    __syncthreads();

    if (i > 0 && i < N - 1 &&
        j > 0 && j < M - 1 &&
        k > 0 && k < K - 1) {
            if(
                (threadIdx.x >= 1 && threadIdx.x < blockDim.x-1) &&
                (threadIdx.y >= 1 && threadIdx.y < blockDim.y-1) &&
                (threadIdx.z >= 1 && threadIdx.z < blockDim.z-1) 
            ){
                output[k * M * N + j * N + i] =
                c0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
                c1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x-1] +
                c2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x+1] +
                c3 * in_s[threadIdx.z][threadIdx.y-1][threadIdx.x] +
                c4 * in_s[threadIdx.z][threadIdx.y+1][threadIdx.x] +
                c5 * in_s[threadIdx.z-1][threadIdx.y][threadIdx.x] +
                c6 * in_s[threadIdx.z+1][threadIdx.y][threadIdx.x];
            }

    } else if (i >= 0 && i < N &&
        j >= 0 && j < M &&
        k >= 0 && k < K ) {
        output[k * M * N + j * N + i] = in_s[threadIdx.z][threadIdx.y][threadIdx.x];
    }
}

void stencil(float *input, float *output, int M, int N, int K) {
    size_t size = M * N * K * sizeof(float);
    float *input_d, *output_d;

    cudaMalloc((void **)&input_d, size);
    cudaMalloc((void **)&output_d, size);

    cudaMemcpy(input_d, input, size, cudaMemcpyHostToDevice);

    dim3 block(INPUT_TILE, INPUT_TILE, INPUT_TILE);
    dim3 grid((N + OUTPUT_TILE - 1) / OUTPUT_TILE,
              (M + OUTPUT_TILE - 1) / OUTPUT_TILE,
              (K + OUTPUT_TILE - 1) / OUTPUT_TILE);

    _stencil<<<grid, block>>>(input_d, output_d, M, N, K);

    cudaMemcpy(output, output_d, size, cudaMemcpyDeviceToHost);

    cudaFree(input_d);
    cudaFree(output_d);
}

void stencil_cpu(const float *input, float *output, int M, int N, int K) {
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < M; ++j) {
            for (int i = 0; i < N; ++i) {
                int idx = k * M * N + j * N + i;

                if (i > 0 && i < N - 1 &&
                    j > 0 && j < M - 1 &&
                    k > 0 && k < K - 1) {
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

int main() {
    int M = 64; // Reduced size for quick testing and verification
    int N = 64;
    int K = 64;
    size_t total = M * N * K;

    float *input = new float[total];
    float *output_gpu = new float[total];
    float *output_cpu = new float[total];

    generateRandomMatrix(input, M, N, K);

    // Run GPU stencil
    stencil(input, output_gpu, M, N, K);

    // Run CPU stencil
    stencil_cpu(input, output_cpu, M, N, K);

    // Compare outputs
    float epsilon = 1e-5f;
    bool mismatch_found = false;
    for (size_t i = 0; i < total; ++i) {
        if (std::fabs(output_gpu[i] - output_cpu[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i
                      << ": GPU = " << std::fixed << std::setprecision(8) << output_gpu[i]
                      << ", CPU = " << output_cpu[i]
                      << ", Diff = " << std::fabs(output_gpu[i] - output_cpu[i]) << std::endl;
            mismatch_found = true;
            break;
        }
    }

    if (!mismatch_found) {
        std::cout << "Verification successful! GPU and CPU outputs match within tolerance." << std::endl;
    } else {
        std::cout << "Verification failed! Outputs differ." << std::endl;
    }

    delete[] input;
    delete[] output_gpu;
    delete[] output_cpu;
    return 0;
}