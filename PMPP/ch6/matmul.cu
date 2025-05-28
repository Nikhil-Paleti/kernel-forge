#include <iostream> 
#include <cuda_runtime.h>
#include <cstdlib> 

#define TILE_WIDTH 16
#define COARSE_FACTOR 2

void generateRandomMatrix(float *matrix, int rows, int cols){
    for(int i=0; i<rows*cols; i++){
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

__global__ void _matmul(float *A, float *B, float *C, int M, int N, int K){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = (blockIdx.x * COARSE_FACTOR) * blockDim.x  + tx;
    int y = blockIdx.y * blockDim.y + ty; 

    float acc[COARSE_FACTOR];
    for (int c=0; c < COARSE_FACTOR; c++){
        acc[c] = 0.0;
    }

    for (int ph=0; ph < (K + TILE_WIDTH - 1) / (TILE_WIDTH); ph++){
        if (ph*TILE_WIDTH + tx < K && y < M ){
            Mds[ty][tx] = A[y*K + ph*TILE_WIDTH + tx];
        }else{
            Mds[ty][tx] = 0.0;
        }

        for(int c=0; c< COARSE_FACTOR; c++){
            if (ph*TILE_WIDTH + ty < K && x + TILE_WIDTH*c < N){
                Nds[ty][tx] = B[(ph*TILE_WIDTH + ty) * N + (x + TILE_WIDTH * c)];
            }else{
                Nds[ty][tx] = 0.0;
            }
            __syncthreads();

            for (int k =0; k < TILE_WIDTH; k++){
                acc[c] += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();

        }

    }
    for (int c=0; c < COARSE_FACTOR; c++){
        if (x + c*TILE_WIDTH < N && y < M){
            C[ y*N + x + c*TILE_WIDTH] = acc[c]; 
        }
    }
}

void matmul(float *A_h, float *B_h, float *C_h, int M, int N, int K){
    size_t size_A = M*K * sizeof(float);
    size_t size_B = K*N * sizeof(float);
    size_t size_C = M*N * sizeof(float);
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, size_A);
    cudaMalloc((void**)&B_d, size_B);
    cudaMalloc((void**)&C_d, size_C);
    cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice);
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH*COARSE_FACTOR -1)/ (TILE_WIDTH*COARSE_FACTOR), (M + TILE_WIDTH -1)/ TILE_WIDTH);
    _matmul<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
    cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(){
    int M = 1024;
    int N = 1024;
    int K = 768; 
    float *A_h = new float[M*K];
    float *B_h = new float[K*N];
    float *C_h = new float[M*N];
    generateRandomMatrix(A_h, M, K);
    generateRandomMatrix(B_h, K, N);
    matmul(A_h, B_h, C_h, M, N, K);
    delete[] A_h;
    delete[] B_h;
    delete[] C_h;
    return 0;
}