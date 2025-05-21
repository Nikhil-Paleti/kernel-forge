#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

void generateRandomMatrix(float* matrix, int rows, int cols){
    for (int i=0; i<rows*cols; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void printMatrix(float* matrix, int rows, int cols) {
    for (int i=0; i< rows; i++){
        for (int j=0; j<cols; j++){
            std::cout<<matrix[i*cols + j]<<  ;
        }
        std::cout<<std::endl;
    }
}

__global__ void matmul(float* A, float* B, float* C, int rows, int cols, int width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < rows) && (col < cols)) {
        float val = 0.0;
        for (int k=0; k<width; k++){
            val += A[row*width + k] * B[k*cols + col];
        }

        C[row*cols + col] = val;
    }
    

}

int main(){
    srand(time(0)); 

    int M = 4, K = 3, N = 5;

    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N]; 

    generateRandomMatrix(A, M, K);
    generateRandomMatrix(B, K, N);

    std::cout<<----------A---------<<std::endl;
    printMatrix(A, M, K);
    std::cout<<----------B---------<<std::endl;
    printMatrix(B, K, N);

    float *A_d, *B_d, *C_d;
    cudaMalloc( (void**)&A_d , M*K*sizeof(float));
    cudaMalloc( (void**)&B_d , K*N*sizeof(float));
    cudaMalloc( (void**)&C_d , M*N*sizeof(float));

    cudaMemcpy(A_d, A, M*K*sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, K*N*sizeof(float) , cudaMemcpyHostToDevice);


    dim3 blockDim(2,2,1);
    dim3 gridDim((N + 1) / 2, (M + 1) / 2, 1);

    matmul<<<gridDim, blockDim>>>(A_d, B_d, C_d, M, N, K);

    cudaMemcpy(C, C_d, M*N*sizeof(float) , cudaMemcpyDeviceToHost);

    std::cout<<----------C---------<<std::endl;
    printMatrix(C, M, N);

    return 0;
}
