#include <iostream>
#include <cuda_runtime.h>

__global__ void vecAddKernel(float* A, float* B, float* C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
    }
}
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {

    int size = n * sizeof(float);

    float *A_d;
    float *B_d;
    float *C_d;

    cudaMalloc( (void**)&A_d , size);
    cudaMalloc( (void**)&B_d, size);
    cudaMalloc( (void**)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    vecAddKernel<<< n/2, 2>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}

int main() {
    int n = 6;
    float A_h[] = {0,1,2,3,4,5};
    float B_h[] = {5,4,3,2,1,0};
    float C_h[n];

    vecAdd(A_h, B_h, C_h, n);
    

    for(int i=0; i<n; i++){
        std::cout<<C_h[i]<<std::endl;
    }

    return 0;
}
