#include <stdio.h>
#include <stdint.h>
#define N_ITEMS 1024*1024

//Kernel para la suma de vectores en CUDA

__global__ void no_registered(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = 0; i < 100; i++){
        b[0] += sinf(b[0]) * cosf(b[0]) + sqrtf(b[0] + 1.0f) + atanf(b[0]);
        if (i < n) {
            c[i] = a[i] + *b;
        }
    }

}

__global__ void registered(float *a, float *b, float *c, int n) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     float temp = *b;
     for (size_t i = 0; i < 100; i++){
        temp += sinf(temp) * cosf(temp) + sqrtf(temp + 1.0f) + atanf(temp);
        if (i < n) {
            c[i] = a[i] + temp;
        }
    }
 
}

int main() {


    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, 0);

    printf("Device %s\n", deviceProp.name);
    printf("Max threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);

    // Alojar memoria en el host
    float *h_a = (float*)malloc(sizeof(float) * N_ITEMS);
    float h_b;
    float *h_c = (float*)malloc(sizeof(float) * N_ITEMS);

    // Inicializar el vectores h_a y h_b
    h_b = (20.0 * rand() / RAND_MAX) - 10;        
    for (int i = 0; i < N_ITEMS; i++) {
        h_a[i] = (20.0 * rand() / RAND_MAX) - 10;        
    }

    // Alojar memoria en el dispositivo
    float *d_a, *d_b, *d_c, milliseconds;
    cudaMalloc((void**)&d_a, sizeof(float) * N_ITEMS);
    cudaMalloc((void**)&d_b, sizeof(float));
    cudaMalloc((void**)&d_c, sizeof(float) * N_ITEMS);

    // Transferir datos desde el host al dispositivo
    cudaMemcpy(d_a, h_a, sizeof(float) * N_ITEMS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice);

    // Configurar los parámetros del kernel
    int blockSize = 1024;
    int gridSize = (N_ITEMS + blockSize - 1) / blockSize;
    
    // Crear eventos para medir el tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Registrar el evento de inicio
    cudaEventRecord(start, 0);

    // Llamar al kernel
    
    no_registered<<<gridSize, blockSize>>>(d_a, d_b, d_c, N_ITEMS);

    // Registrar el evento de fin
    cudaEventRecord(stop, 0);

    // Sincronizar para asegurar que el kernel haya terminado
    cudaEventSynchronize(stop);

    
    // Transferir resultado desde el dispositivo al host
    cudaMemcpy(h_c, d_c, sizeof(float) * N_ITEMS, cudaMemcpyDeviceToHost);
    // Medir el tiempo transcurrido
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Tiempo de ejecución del kernel sin registro: %.4f ms\n", milliseconds);

    cudaEventRecord(start, 0);

    registered<<<gridSize, blockSize>>>(d_a, d_b, d_c, N_ITEMS);

    // Registrar el evento de fin
    cudaEventRecord(stop, 0);

    // Sincronizar para asegurar que el kernel haya terminado
    cudaEventSynchronize(stop);
    
    // Transferir resultado desde el dispositivo al host
    cudaMemcpy(h_c, d_c, sizeof(float) * N_ITEMS, cudaMemcpyDeviceToHost);
    // Medir el tiempo transcurrido
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Tiempo de ejecución del kernel con registro: %.4f ms\n", milliseconds);

    // Liberar memoria
    free(h_a);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Destruir eventos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
