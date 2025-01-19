#include <iostream>
#include <cuda_runtime.h>

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    for (int i=0; i<n; ++i){
        C_h[i] = A_h[i] + B_h[i];
    }
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
