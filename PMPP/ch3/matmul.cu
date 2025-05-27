#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

void createRandomMatrix(float *matrix, int rows, int cols){
    for(int i =0; i< rows*cols; i++){
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

__global__ void _matmul_kernel(float *a, float *b, float *c, int M, int N, int K){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    if ((x < M ) && (y < N) ) {
        float acc = 0.0f; 
        for (int k =0; k<K; k++){
            acc += a[y*K + k] * b[N*k + x];
        }
        c[y*N + x] = acc; 
    }

}
void matmul(float *a, float *b, float *c, int m, int n, int k){
    int sike_a = m*k * sizeof(float);
    int sike_b = k*n * sizeof(float);
    int sike_c = m*n * sizeof(float);
    float *a_d, *b_d, *c_d; 
    cudaMalloc((void**)&a, sike_a);
    cudaMalloc((void**)&b, sike_b);
    cudaMalloc((void**)&c, sike_c);
    cudaMemcpy(a_d, a, sike_a, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sike_b, cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid( (n + block.x - 1) / block.x , (m+ block.y - 1)/ block.y );
    _matmul_kernel<<<grid, block>>>(a_d, b_d, c_d, m, n, k);
    cudaMemcpy(c, c_d, sike_c, cudaMemcpyDeviceToHost);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

int main(){
    int m = 1024;
    int n = 2048;
    int k = 512; 
    float *a = new float[m*k];
    float *b = new float[k*n];
    float *c = new float[m*n];
    createRandomMatrix(a, m, k);
    createRandomMatrix(b, k, n);
    matmul(a, b, c, m, n, k);
    delete[] a;
    delete[] b; 

    return 0;
}
