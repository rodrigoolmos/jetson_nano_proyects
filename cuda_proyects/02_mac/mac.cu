#include <stdio.h>
#include <stdint.h>

// Número de elementos en los vectores < blockSize
// la sincronizacion solo se puede hacer en un bloque
#define N_ITEMS (1023) 


// Kernel para MAC en CUDA
__global__ void mac_naive(float *a, float *b, float *c, float *tmp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        tmp[i] = a[i] * b[i];

        __syncthreads();

        if (i == 0) {
            c[0] = 0.0f;
            for (int j = 0; j < n; j++) {
                c[0] += tmp[j];
            }
        }
    }
}

// Kernel para MAC en CUDA
__global__ void mac(float *a, float *b, float *c, float *tmp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Cálculo inicial: cada hilo multiplica un elemento
    if (i < n) {
        tmp[i] = a[i] * b[i];
        __syncthreads();

        // Añadir ultimo elemnto si no es par
        if (n%2 == 1 && n > 2){
            if (i == 0){
                tmp[0] += tmp[n-1];
            }
        }
        __syncthreads();

        // Reducción en bloque: se asume que n es potencia de 2 y blockDim.x == n
        for (int stride = n / 2; stride > 0; stride /= 2) {
            if (i < stride){
                if(i == 0) printf("stride: %d,\t tmp[i + stride]: %d,\t tmp[i]: %f\n",stride, tmp[i + stride], tmp[i]);
                tmp[i] += tmp[i + stride];
            }
            __syncthreads();

            // Añadir ultimo elemnto si no es par
            if (stride%2 == 1 && stride > 2){
                if (i == 0){
                    tmp[0] += tmp[stride - 1];
                }
            }
            __syncthreads();
        }

        // Solo el hilo 0 del bloque escribe el resultado final
        if (i == 0) {
            *c = tmp[0];
        }
    }
}


float mac(float *a, float *b, int n) {
    float result = 0.0f;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

int main() {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceProp deviceProp;

    int blockSize;
    int gridSize;

    float milliseconds = 0;

    float *d_a, *d_b, *d_tmp, *d_c, result_cpu;

    // Alojar memoria en el host
    float *h_a = (float*)malloc(sizeof(float) * N_ITEMS);
    float *h_b = (float*)malloc(sizeof(float) * N_ITEMS);
    float h_c;

    cudaGetDeviceProperties(&deviceProp, 0);

    printf("Device %s\n", deviceProp.name);
    printf("Max threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);

    // Inicializar los vectores h_a y h_b
    for (int i = 0; i < N_ITEMS; i++) {
        h_a[i] = 1;//2.0 * rand() / RAND_MAX - 1;
        h_b[i] = 1;//2.0 * rand() / RAND_MAX - 1;
    }

    // Alojar memoria en el dispositivo
    cudaMalloc((void**)&d_a, sizeof(float) * N_ITEMS);
    cudaMalloc((void**)&d_b, sizeof(float) * N_ITEMS);
    cudaMalloc((void**)&d_tmp, sizeof(float) * N_ITEMS);
    cudaMalloc((void**)&d_c, sizeof(float));

    // Transferir datos desde el host al dispositivo
    cudaMemcpy(d_a, h_a, sizeof(float) * N_ITEMS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * N_ITEMS, cudaMemcpyHostToDevice);

    // Configurar los parámetros del kernel
    blockSize = 1024;
    gridSize = (N_ITEMS + blockSize - 1) / blockSize;

    cudaEventRecord(start, 0);    
    // Llamar al kernel
    mac<<<gridSize, blockSize>>>(d_a, d_b, d_c, d_tmp, N_ITEMS);
    cudaEventRecord(stop, 0);

    // Transferir resultado desde el dispositivo al host
    cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    result_cpu = mac(h_a, h_b, N_ITEMS);
    printf("Resultado de MAC -> (CPU %f, GPU %f)\n", result_cpu, h_c);

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo de ejecución del kernel: %.4f ms\n", milliseconds);

    // Liberar memoria
    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
