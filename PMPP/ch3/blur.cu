#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

void createRandomImage(float *image, int width, int height){
    for (int i=0; i<width*height; i++){
        image[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

__global__ void _blurKernel(float *image, int width, int height, int radius){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float acc = 0.0f;
    int count = 0;
    for (int i=-radius; i<radius+1; i++){
        for (int j=-radius; j<radius+1; j++){
            int nx = x + i;
            int ny = y + j;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height){
                int index = ny * width + nx;
                acc += image[index];
                count++;
            }
        }
    }
    if (count > 0){
        image[y*width+x] = acc / count;
    }
}

void blur(float *image, int width, int height, int radius){
    int size = width*height*sizeof(float);
    float *d_image;
    cudaMalloc((void**)&d_image, size);
    cudaMemcpy(d_image, image, size, cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
    _blurKernel<<<grid, block>>>(d_image, width, height, radius);
    cudaMemcpy(image, d_image, size, cudaMemcpyDeviceToHost);
    cudaFree(d_image);
}

int main(){
    int width = 1024;
    int height = 1024;
    int radius = 1;
    float *h_image = new float[width*height];
    createRandomImage(h_image, width, height);
    blur(h_image, width, height, radius);
    delete[] h_image;

    return 0;
}