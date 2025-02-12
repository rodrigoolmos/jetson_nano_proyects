#include <stdio.h>
#define N_ITEMS 1024*1024


// Kernel para la suma de vectores en CUDA
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceProp deviceProp;

    int blockSize;
    int gridSize;

    float milliseconds = 0;

    float *d_a, *d_b, *d_c;

    // Alojar memoria en el host
    float *h_a = (float*)malloc(sizeof(float) * N_ITEMS);
    float *h_b = (float*)malloc(sizeof(float) * N_ITEMS);
    float *h_c = (float*)malloc(sizeof(float) * N_ITEMS);

    cudaGetDeviceProperties(&deviceProp, 0);

    printf("Device %s\n", deviceProp.name);
    printf("Max threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);

    // Inicializar los vectores h_a y h_b
    for (int i = 0; i < N_ITEMS; i++) {
        h_a[i] = 2.0f*i;
        h_b[i] = 3.0f+i;
    }

    // Alojar memoria en el dispositivo
    cudaMalloc((void**)&d_a, sizeof(float) * N_ITEMS);
    cudaMalloc((void**)&d_b, sizeof(float) * N_ITEMS);
    cudaMalloc((void**)&d_c, sizeof(float) * N_ITEMS);

    // Transferir datos desde el host al dispositivo
    cudaMemcpy(d_a, h_a, sizeof(float) * N_ITEMS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * N_ITEMS, cudaMemcpyHostToDevice);

    // Configurar los parámetros del kernel
    blockSize = 1024;
    gridSize = (N_ITEMS + blockSize - 1) / blockSize;

    cudaEventRecord(start, 0);    
    // Llamar al kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N_ITEMS);
    cudaEventRecord(stop, 0);

    // Transferir resultado desde el dispositivo al host
    cudaMemcpy(h_c, d_c, sizeof(float) * N_ITEMS, cudaMemcpyDeviceToHost);

    printf("Resultado de la suma:\n");
    for (int i = 0; i < 100; i++) {
        printf("%.2f + %.2f = %.2f\n", h_a[i], h_b[i], h_c[i]);
    }

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo de ejecución del kernel: %.4f ms\n", milliseconds);

    // Liberar memoria
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
