#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib> 
#include <cmath> // For fabs
#include <vector> // For storing CPU output if you prefer
#include <iomanip> // For std::fixed and std::setprecision

#define FILTER_RADIUS 2 
#define TILE_DIM 32

__constant__ float F[(2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1)];

void generateRandomMatrix(float *matrix, int rows, int cols, int seed=0){
    srand(seed);
    for (int i =0; i < rows*cols; i++){
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

__global__ void _conv2d(float* N, float *P ,int height, int width, int r){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float N_s[TILE_DIM][TILE_DIM];
    if( (x < width) && (y < height) ){
        N_s[threadIdx.y][threadIdx.x] = N[y*width + x];
    }else{
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    if( (x < width) && (y < height) ){
        float acc = 0.0;
        for (int j=0; j<2*r+1; j++){
            for(int i=0; i<2*r+1; i++){
                if (
                     ((ty - r + j) >= 0 && (ty - r + j) < TILE_DIM) && 
                     ((tx - r + i) >= 0 && (tx - r + i) < TILE_DIM)
                ){
                    acc +=  F[j*(2*r+1) + i] * N_s[ty - r+j][tx - r+i];
                }else{
                    // fetch from global memory
                    int global_x = x - r + i;
                    int global_y = y - r + j;
                    if ((global_x >= 0 && global_x < width) && (global_y >=0 && global_y < height)) {
                        acc +=  F[j*(2*r+1) + i] * N[global_y*width + global_x];
                    }
                }
            }
        }
        P[y*width + x] = acc;
    }
}

void conv2d(float *image, float *filter, float *output, int M, int N, int r){
    size_t image_size = M*N * sizeof(float);
    size_t filter_size = (2*r+1) * (2*r+1) * sizeof(float);
    float *image_d, *output_d;
    cudaMalloc((void**)&image_d, image_size);
    cudaMalloc((void**)&output_d, image_size);
    cudaMemcpy(image_d, image, image_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, filter, filter_size);
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM , (M+TILE_DIM-1)/TILE_DIM);
    _conv2d<<<grid, block>>>(image_d, output_d, M, N, r);
    cudaMemcpy(output, output_d, image_size, cudaMemcpyDeviceToHost);
    cudaFree(image_d);
    cudaFree(output_d);
}

void conv2d_cpu(const float *image, const float *filter, float *output, int height, int width, int r) {
    int filter_dim = 2 * r + 1;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float acc = 0.0f;
            for (int j = 0; j < filter_dim; ++j) { // Filter row
                for (int i = 0; i < filter_dim; ++i) { // Filter col
                    int image_x = x - r + i;
                    int image_y = y - r + j;

                    if ((image_x >= 0 && image_x < width) && (image_y >= 0 && image_y < height)) {
                        acc += image[image_y * width + image_x] * filter[j * filter_dim + i];
                    }
                }
            }
            output[y * width + x] = acc;
        }
    }
}

int main(){
    int M = 1024;
    int N = 1024;
    int r = FILTER_RADIUS;
    int filter_dim_actual = (2*r + 1); // Renamed from filter_size to avoid conflict
    float *image = new float[M*N];
    float *filter = new float[ filter_dim_actual * filter_dim_actual];
    float *output_gpu = new float[M*N]; // Output from GPU
    float *output_cpu = new float[M*N]; // Output from CPU

    generateRandomMatrix(image, M, N, 0); // Use fixed seeds for reproducibility
    generateRandomMatrix(filter, filter_dim_actual, filter_dim_actual, 1);

    // GPU Convolution
    conv2d(image, filter, output_gpu, M, N, r);

    // CPU Convolution
    conv2d_cpu(image, filter, output_cpu, M, N, r);

    // Comparison
    float epsilon = 1e-5f; // Tolerance for floating point comparison
    bool mismatch_found = false;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(output_gpu[i] - output_cpu[i]) > epsilon) {
            std::cerr << "Mismatch found at index " << i << "!" << std::endl;
            std::cerr << "GPU output: " << std::fixed << std::setprecision(8) << output_gpu[i] << std::endl;
            std::cerr << "CPU output: " << std::fixed << std::setprecision(8) << output_cpu[i] << std::endl;
            std::cerr << "Difference: " << std::fabs(output_gpu[i] - output_cpu[i]) << std::endl;
            mismatch_found = true;
            // You might want to break here or limit the number of printed mismatches
            break; 
        }
    }

    if (!mismatch_found) {
        std::cout << "Verification successful! GPU and CPU outputs match within tolerance." << std::endl;
    } else {
        std::cout << "Verification failed! GPU and CPU outputs differ." << std::endl;
    }

    delete[] image;
    delete[] filter;
    delete[] output_gpu;
    delete[] output_cpu;
    return 0;
}