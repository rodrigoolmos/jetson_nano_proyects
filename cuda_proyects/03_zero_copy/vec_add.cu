#include <stdio.h>
#define N_ITEMS 1024*1024*128


// Kernel para la suma de vectores en CUDA
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vector_add(float *a, float *b, float *c, int n){

    cudaEvent_t start1, stop1;
    float milliseconds = 0;

    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1, 0);    
    for (int i = 0; i < n; i++){
        c[i] = a[i] + b[i];
    }
    cudaEventRecord(stop1, 0);
 
    cudaEventElapsedTime(&milliseconds, start1, stop1);
    printf("Tiempo de ejecución CLP: %.4f ms\n", milliseconds);

}

int main() {

    cudaEvent_t start1, stop1;
    cudaEvent_t start2, stop2;

    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);
    cudaDeviceProp deviceProp;

    int blockSize;
    int gridSize;

    float milliseconds1 = 0;
    float milliseconds2 = 0;


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
        h_a[i] = i;
        h_b[i] = i;
    }

    // Alojar memoria en el dispositivo
    cudaMalloc((void**)&d_a, sizeof(float) * N_ITEMS);
    cudaMalloc((void**)&d_b, sizeof(float) * N_ITEMS);
    cudaMalloc((void**)&d_c, sizeof(float) * N_ITEMS);

    // Configurar los parámetros del kernel
    blockSize = 1024;
    gridSize = (N_ITEMS + blockSize - 1) / blockSize;

    // Transferir datos desde el host al dispositivo
    cudaEventRecord(start1, 0);    
    cudaMemcpy(d_a, h_a, sizeof(float) * N_ITEMS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * N_ITEMS, cudaMemcpyHostToDevice);
    // Llamar al kernel
    cudaEventRecord(start2, 0);    
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N_ITEMS);
    cudaEventRecord(stop2, 0);
    // Transferir resultado desde el dispositivo al host
    cudaMemcpy(h_c, d_c, sizeof(float) * N_ITEMS, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop1, 0);

    printf("Resultado de la suma:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f + %.2f = %.2f\n", h_a[i], h_b[i], h_c[i]);
    }

    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    cudaEventElapsedTime(&milliseconds2, start2, stop2);

    printf("Tiempo de ejecución del kernel: %.4f ms tiempo total  %.4f ms\n", milliseconds2, milliseconds1);

    vector_add(h_a, h_b, h_c, N_ITEMS);

    // Liberar memoria
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
