#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 1024*2  // Debe ser múltiplo de 16 para usar Tensor Cores

// Kernel de multiplicación de matrices en GPU
__global__ void matmul_gpu(float *d_A, float *d_B, float *d_C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N){
        for (int k = 0; k < N; k++) {
           d_C[i * N + j] += d_A[i * N + k] * d_B[k * N + j];
        }
    }
}

// Función para imprimir matrices
void printMatrix(const float* mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

// Multiplicación de matrices en CPU
void matmul_cpu(float *h_A, float *h_B, float *h_C_cpu, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < cols; k++) {
                sum += h_A[i * cols + k] * h_B[k * cols + j];
            }
            h_C_cpu[i * cols + j] = sum;
        }
    }
}

int main() {
    struct timeval start, stop;
    double elapsed;
    int i, j, error = 0;

    float *d_A, *d_B, *d_C;  // Punteros en GPU
    float *h_A, *h_B, *h_C, *h_C_cpu;  // Punteros en CPU

    // Alojar memoria en la CPU
    h_A = (float*)malloc(N * N * sizeof(float));
    h_B = (float*)malloc(N * N * sizeof(float));
    h_C = (float*)malloc(N * N * sizeof(float));
    h_C_cpu = (float*)malloc(N * N * sizeof(float));

    // Inicializar matrices en la CPU
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (rand() % 20 - 10) / 10.0f;
        h_B[i] = (rand() % 20 - 10) / 10.0f;
        h_C[i] = 0.0f;
        h_C_cpu[i] = 0.0f;
    }

    // Alojar memoria en la GPU
    cudaMalloc((void**)&d_A, sizeof(float) * N * N);
    cudaMalloc((void**)&d_B, sizeof(float) * N * N);
    cudaMalloc((void**)&d_C, sizeof(float) * N * N);

    
    // Definir tamaño del bloque y la cuadrícula en 2D
    dim3 blockSize(32, 32);  // Bloques de 32x32 hilos (máximo 1024 hilos por bloque)
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
    (N + blockSize.y - 1) / blockSize.y);
    
    gettimeofday(&start, NULL); // Iniciar medición de tiempo
    // Copiar datos desde CPU a GPU
    cudaMemcpy(d_A, h_A, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    
    // Ejecutar kernel en GPU
    matmul_gpu<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    // Esperar a que termine
    cudaDeviceSynchronize();  
    // Copiar resultado de vuelta a la CPU
    cudaMemcpy(h_C, d_C, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

    gettimeofday(&stop, NULL); // Finalizar medición de tiempo

    // Calcular tiempo en milisegundos
    elapsed = (stop.tv_sec - start.tv_sec) * 1000.0;
    elapsed += (stop.tv_usec - start.tv_usec) / 1000.0;

    printf("Tiempo de ejecución en la GPU: %.4f ms tamaño %ix%i\n", elapsed, N, N);

    // Calcular resultado en CPU (solo para verificar)
    gettimeofday(&start, NULL);
    matmul_cpu(h_A, h_B, h_C_cpu, N, N);
    gettimeofday(&stop, NULL);

    elapsed = (stop.tv_sec - start.tv_sec) * 1000.0;
    elapsed += (stop.tv_usec - start.tv_usec) / 1000.0;

    // Verificación de resultados
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (fabs(h_C[i * N + j] - h_C_cpu[i * N + j]) > 0.1f) {
                error = 1;
                printf("%f, %f\n", h_C[i * N + j], h_C_cpu[i * N + j]);
            }
        }
    }

    printf("Tiempo de ejecución CPU:  %.4f ms tamaño %ix%i\n", elapsed, N, N);
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
