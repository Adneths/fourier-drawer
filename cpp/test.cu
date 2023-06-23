#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>

#define BLOCK_SIZE 16
__global__ void cumsum(float* in, float* out, int n) {
    __shared__ float sBlock[BLOCK_SIZE];

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int tx = threadIdx.x;

    if (id < n)
        sBlock[tx] = in[id];
    else
        sBlock[tx] = 0.0f;
    __syncthreads();

    for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        int index = (tx + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE)
            sBlock[index] += sBlock[index - stride];
        __syncthreads();
    }

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        int index = (tx + 1) * stride * 2 - 1;
        if (index + stride < BLOCK_SIZE)
            sBlock[index+stride] += sBlock[index];
        __syncthreads();
    }

    if (id < n)
        out[id] = sBlock[tx];
}

int main() {
    int n = 16;
    float* in_array, * out_array;
    float* d_in_array, * d_out_array;

    // Allocate memory on the host and device
    in_array = (float*)malloc(n * sizeof(float));
    out_array = (float*)malloc(n * sizeof(float));
    cudaMalloc(&d_in_array, n * sizeof(float));
    cudaMalloc(&d_out_array, n * sizeof(float));

    // Initialize input array on host
    for (int i = 0; i < n; i++) {
        in_array[i] = i;
    }
    for (int i = 0; i < n; i++) {
        printf("%.0f ", in_array[i]);
    }
    printf("\n");

    // Copy input array to device
    cudaMemcpy(d_in_array, in_array, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to compute cumulative sum
    cumsum << <(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_in_array, d_out_array, n);

    // Copy result back to host
    cudaMemcpy(out_array, d_out_array, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        printf("%.0f ", out_array[i]);
    }
    printf("\n");

    // Free memory on device and host
    cudaFree(d_in_array);
    cudaFree(d_out_array);
    free(in_array);
    free(out_array);

    return 0;
}