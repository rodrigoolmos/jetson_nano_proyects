#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda_runtime.h>

#define N (1024*2)
#define TILE_SIZE 32

__global__ void matmul_gpu(float *d_A, float *d_B, float *d_C) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // i=0,1N,2N,3N,4N,5N,6N,7N,8N,9N...N^2-N
    int j = blockIdx.x * blockDim.x + threadIdx.x; // j=0,1,2,3,4,5,6,7,8,9...N-1
    
    __shared__ float sum[N];
    
    if(i<N && j<N){
        sum[j] = 0.0f;
        __syncthreads();
        // Acumular las sumas parciales en el vector partial
        for (int k = 0; k < N; k++) {
            sum[j] += d_A[i * N + k] * d_B[k * N + j];
        }
        d_C[i * N + j] = sum[j];
    }
}

void printMatrix(const float* mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

// Multiplicación de matrices en CPU (implementación sencilla)
void matmul_cpu(float *h_A, float *h_B, float *h_C_cpu) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += h_A[i * N + k] * h_B[k * N + j];
            }
            h_C_cpu[i * N + j] = sum;
        }
    }
}

int main() {
    struct timeval start, stop;
    double elapsed;
    int error = 0;

    size_t bytes = N * N * sizeof(float);

    // Alojar memoria en la CPU
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);

    // Inicializar matrices en la CPU
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (rand() % 20 - 10) / 10.0f;
        h_B[i] = (rand() % 20 - 10) / 10.0f;
        h_C[i] = 0.0f;
        h_C_cpu[i] = 0.0f;
    }

    // Alojar memoria en la GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    
    // Definir el tamaño del bloque y la cuadrícula en 2D
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    gettimeofday(&start, NULL);
    // Copiar datos desde la CPU a la GPU
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    // Ejecutar kernel en GPU
    matmul_gpu<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    // Copiar el resultado de vuelta a la CPU
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    gettimeofday(&stop, NULL);

    elapsed = (stop.tv_sec - start.tv_sec) * 1000.0 +
              (stop.tv_usec - start.tv_usec) / 1000.0;
    printf("Tiempo de ejecución en la GPU: %.4f ms, tamaño %dx%d\n", elapsed, N, N);


    // Calcular el resultado en CPU (solo para verificar)
    gettimeofday(&start, NULL);
    matmul_cpu(h_A, h_B, h_C_cpu);
    gettimeofday(&stop, NULL);
    elapsed = (stop.tv_sec - start.tv_sec) * 1000.0 +
              (stop.tv_usec - start.tv_usec) / 1000.0;
    printf("Tiempo de ejecución en la CPU: %.4f ms, tamaño %dx%d\n", elapsed, N, N);

    // Verificación de resultados
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(h_C[i * N + j] - h_C_cpu[i * N + j]) > 0.1f) {
                error = 1;
                printf("Desajuste en [%d][%d]: GPU = %f, CPU = %f\n", i, j,
                       h_C[i * N + j], h_C_cpu[i * N + j]);
            }
        }
    }
    printf("Multiplicación de matrices %s\n", error ? "incorrecta" : "correcta");

    // Liberar memoria en GPU y CPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);

    return 0;
}
