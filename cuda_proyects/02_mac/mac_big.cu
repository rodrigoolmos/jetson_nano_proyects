#include <stdio.h>
#include <stdint.h>


#define N_ITEMS (1024*1024+123) 

// Kernel para MAC en CUDA
__global__ void mul(float *a, float *b, float *tmp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Cálculo inicial: cada hilo multiplica un elemento
    if (i < n) {
        tmp[i] = a[i] * b[i];
    }
}

__global__ void acu(float *tmp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
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
    int i;
    int remaining;

    float milliseconds = 0;

    float *d_a, *d_b, *d_tmp, result_cpu;

    float *h_a = (float*)malloc(sizeof(float) * N_ITEMS);
    float *h_b = (float*)malloc(sizeof(float) * N_ITEMS);
    float h_c = 0;
    float h_c_acum = 0;


    cudaGetDeviceProperties(&deviceProp, 0);

    printf("Device %s\n", deviceProp.name);
    printf("Max threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);

    for (i = 0; i < N_ITEMS; i++) {
        h_a[i] = 2.0 * rand() / RAND_MAX - 1;
        h_b[i] = 2.0 * rand() / RAND_MAX - 1;
    }

    cudaMalloc((void**)&d_a, sizeof(float) * N_ITEMS);
    cudaMalloc((void**)&d_b, sizeof(float) * N_ITEMS);
    cudaMalloc((void**)&d_tmp, sizeof(float) * N_ITEMS);

    cudaMemcpy(d_a, h_a, sizeof(float) * N_ITEMS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * N_ITEMS, cudaMemcpyHostToDevice);

    blockSize = 1024;
    gridSize = (N_ITEMS + blockSize - 1) / blockSize;

    cudaEventRecord(start, 0);    
    // Llamar al kernel
    mul<<<gridSize, blockSize>>>(d_a, d_b, d_tmp, N_ITEMS);
    cudaDeviceSynchronize(); // Asegurarse de que el kernel terminó
    for (i = 0; i < N_ITEMS/blockSize; i++){
        acu<<<1, blockSize>>>(d_tmp + i*blockSize, blockSize);
        cudaDeviceSynchronize(); // Asegurarse de que el kernel terminó
        cudaMemcpy(&h_c, d_tmp + i*blockSize, sizeof(float), cudaMemcpyDeviceToHost);
        h_c_acum += h_c;
    }
    result_cpu = mac(h_a, h_b, N_ITEMS);
    printf("Resultado de MAC -> (CPU %f, GPU %f)\n", result_cpu, h_c_acum);

    remaining = N_ITEMS%blockSize;
    if(remaining != 0){
        acu<<<1, remaining>>>(d_tmp + i*blockSize, remaining);
        cudaDeviceSynchronize(); // Asegurarse de que el kernel terminó
        cudaMemcpy(&h_c, d_tmp + i*blockSize, sizeof(float), cudaMemcpyDeviceToHost);
        h_c_acum += h_c;        
    }

    cudaEventRecord(stop, 0);

    printf("Resultado de MAC -> (CPU %f, GPU %f)\n", result_cpu, h_c_acum);

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo de ejecución del kernel: %.4f ms\n", milliseconds);

    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_tmp);

    return 0;
}
