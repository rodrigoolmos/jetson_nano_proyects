#include <stdio.h>

// Kernel para la suma de vectores en CUDA
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1024; // Tamaño de los vectores
    size_t bytes = n * sizeof(float);

    // Alojar memoria en el host
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Inicializar los vectores h_a y h_b
    for (int i = 0; i < n; i++) {
        h_a[i] = 2.0f*i;
        h_b[i] = 3.0f+i;
    }

    // Alojar memoria en el dispositivo
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    // Transferir datos desde el host al dispositivo
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Configurar los parámetros del kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Llamar al kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Transferir resultado desde el dispositivo al host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Imprimir resultados (opcional)
    printf("Resultado de la suma:\n");
    for (int i = 0; i < n; i++) {
        printf("%.2f + %.2f = %.2f\n", h_a[i], h_b[i], h_c[i]);
    }

    // Liberar memoria
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
