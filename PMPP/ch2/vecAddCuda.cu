#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void vecAddKernel(float *a_d, float *b_d, float *c_d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c_d[i] = a_d[i] + b_d[i];
    }
}

void add(float *a_h, float *b_h, float *c_h, int n) {
    int size = n * sizeof(float);
    float *a_d, *b_d, *c_d;

    cudaMalloc((void**)&a_d, size);
    cudaMalloc((void**)&b_d, size);
    cudaMalloc((void**)&c_d, size);

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    vecAddKernel<<<numBlocks, blockSize>>>(a_d, b_d, c_d, n);
    cudaDeviceSynchronize(); // ensure completion

    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

void generateRandomVector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = static_cast<int>(rand() % 100); // integers between 0–99
    }
}

int main() {
    int n;
    std::cout << "Enter n: ";
    std::cin >> n;

    float* a_h = new float[n];
    float* b_h = new float[n];
    float* c_h = new float[n];

    generateRandomVector(a_h, n);
    generateRandomVector(b_h, n);

    add(a_h, b_h, c_h, n);

    for (int i = 0; i < n; i++) {
        std::cout << a_h[i] << " + " << b_h[i] << " = " << c_h[i] << std::endl;
    }

    delete[] a_h;
    delete[] b_h;
    delete[] c_h;

    return 0;
}
