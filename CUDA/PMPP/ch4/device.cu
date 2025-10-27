#include <iostream>
#include <cuda_runtime.h>

int main(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i=0; i<deviceCount; i++){
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        std::cout << "--------------------------------" << std::endl;
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
        std::cout << "Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Device Multiprocessor Count: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "Shared Memory per Block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "Registers per Block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
        std::cout << "Max Threads per SM: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Max regs per SM: " << deviceProp.regsPerMultiprocessor << std::endl;
        std::cout << "Max shared memory per SM: " << deviceProp.sharedMemPerMultiprocessor << " bytes" << std::endl;
        std::cout << "Clock Rate: " << deviceProp.clockRate << " kHz" << std::endl;
        std::cout << "Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }
    return 0;
}