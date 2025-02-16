#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N_ITEMS 1024*1024*128


// Kernel para la suma de vectores en CUDA
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vector_add(float *a, float *b, float *c, int n) {
    struct timeval start, stop;
    double elapsed;

    gettimeofday(&start, NULL); // Iniciar medición de tiempo

    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }

    gettimeofday(&stop, NULL); // Finalizar medición de tiempo

    // Calcular tiempo en milisegundos
    elapsed = (stop.tv_sec - start.tv_sec) * 1000.0;      // Segundos a milisegundos
    elapsed += (stop.tv_usec - start.tv_usec) / 1000.0;   // Microsegundos a milisegundos

    printf("Tiempo de ejecución CPU: %.4f ms\n", elapsed);
}

int main() {

    cudaEvent_t start1, stop1;

    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaDeviceProp deviceProp;

    int blockSize;
    int gridSize;

    float milliseconds1 = 0;

    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c;

    // Alojar memoria en el host
    cudaHostAlloc((void**)&h_a, sizeof(float) * N_ITEMS, cudaHostAllocMapped);
    cudaHostAlloc((void**)&h_b, sizeof(float) * N_ITEMS, cudaHostAllocMapped);
    cudaHostAlloc((void**)&h_c, sizeof(float) * N_ITEMS, cudaHostAllocMapped);

    cudaGetDeviceProperties(&deviceProp, 0);

    printf("Device %s\n", deviceProp.name);
    printf("Max threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);

    // Inicializar los vectores h_a y h_b
    for (int i = 0; i < N_ITEMS; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    cudaHostGetDevicePointer(&d_a, h_a, 0);
    cudaHostGetDevicePointer(&d_b, h_b, 0);
    cudaHostGetDevicePointer(&d_c, h_c, 0);

    // Configurar los parámetros del kernel
    blockSize = 1024;
    gridSize = (N_ITEMS + blockSize - 1) / blockSize;

    // Transferir datos desde el host al dispositivo
    cudaEventRecord(start1, 0);    
    // Llamar al kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N_ITEMS);
    // Transferir resultado desde el dispositivo al host
    cudaDeviceSynchronize();
    cudaEventRecord(stop1, 0);

    printf("Resultado de la suma:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f + %.2f = %.2f\n", h_a[i], h_b[i], h_c[i]);
    }

    cudaEventElapsedTime(&milliseconds1, start1, stop1);

    printf("Tiempo de ejecución del kernel total %.4f ms\n", milliseconds1);

    vector_add(h_a, h_b, h_c, N_ITEMS);

    // Liberar memoria
    cudaFree(h_a);
    cudaFree(h_b);
    cudaFree(h_c);

    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    return 0;
}
