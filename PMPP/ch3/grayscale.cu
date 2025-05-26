#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

void createRandomImage(float *image, int width, int height){
    for (int i=0; i<width*height; i++){
        image[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

__global__ void _grayscalekernel(float *image, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int index = y * width + x; 
    float r = image[index * 3];
    float g = image[index * 3 + 1];
    float b = image[index * 3 + 2];
    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    image[index] = gray;
}

void grayscale(float *image, int height, int width) {
    int size = width*height*sizeof(float);
    float *d_image;
    cudaMalloc( (void**)&d_image, size); 
    cudaMemcpy(d_image, image, size, cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
    _grayscalekernel<<<grid, block>>>(d_image, width, height);
    cudaMemcpy(image, d_image, size, cudaMemcpyDeviceToHost);
    cudaFree(d_image);
}

int main(){
    int width = 1024;
    int height = 1024; 
    float *h_image = new float[width*height];
    createRandomImage(h_image, width, height);
    grayscale(h_image, width, height);
    delete[] h_image; 
    return 0;
}