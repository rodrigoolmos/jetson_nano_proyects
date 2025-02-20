#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 1024*2  // Tamaño de la matriz (puedes cambiar este valor)

// Función para imprimir una matriz
void printMatrix(const float* mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

void matmul_cpu(float *h_A, float *h_B, float *h_C_cpu, int rows, int cols) {
    int i, j, k;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            float sum = 0.0f;
            for (k = 0; k < cols; k++) {
                sum += h_A[i * cols + k] * h_B[k * cols + j];
            }
            h_C_cpu[i * cols + j] = sum;
        }
    }
}


int main() {
    struct timeval start, stop;
    double elapsed;
    int i, j;

    
    // Asignar memoria en CPU dinámicamente en base a N
    float *h_A = (float*)malloc(N * N * sizeof(float));
    float *h_B = (float*)malloc(N * N * sizeof(float));
    float *h_C = (float*)malloc(N * N * sizeof(float));
    float *h_C_cpu = (float*)malloc(N * N * sizeof(float));

    int error = 0;

    //C=α⋅A⋅B+β⋅C
    float alpha = 1.0f, beta = 0.0f;

    // Inicializar las matrices en CPU
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 20 - 10;         // Valores secuenciales en A
        h_B[i] = rand() % 20 - 10;     // Valores decrecientes en B
        h_C[i] = 0;
        h_C_cpu[i] = 0;
    }

    // Asignar memoria en GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // Copiar matrices desde CPU a GPU
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Inicializar cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);


    gettimeofday(&start, NULL); // Iniciar medición de tiempo
    
    // Multiplicación de matrices: C = A * B
    cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N, 
                N, N, N, 
                &alpha, 
                d_A, N, 
                d_B, N, 
                &beta, 
                d_C, N);

    // Copiar resultado de vuelta a CPU
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    gettimeofday(&stop, NULL); // Finalizar medición de tiempo

    // Calcular tiempo en milisegundos
    elapsed = (stop.tv_sec - start.tv_sec) * 1000.0;      // Segundos a milisegundos
    elapsed += (stop.tv_usec - start.tv_usec) / 1000.0;   // Microsegundos a milisegundos

    printf("Tiempo de ejecución libreria CUDA implementacion GPU: %.4f ms tanmaño %ix%i\n", elapsed, N, N);

    gettimeofday(&start, NULL); // Iniciar medición de tiempo
    matmul_cpu(h_B, h_A, h_C_cpu, N, N);
    gettimeofday(&stop, NULL); // Finalizar medición de tiempo
    
    // Calcular tiempo en milisegundos
    elapsed = (stop.tv_sec - start.tv_sec) * 1000.0;      // Segundos a milisegundos
    elapsed += (stop.tv_usec - start.tv_usec) / 1000.0;   // Microsegundos a milisegundos

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (h_C[i * N + j] != h_C_cpu[i * N + j]){
                error = 1;
            }
        }
    }

    printf("Tiempo de ejecución CPU:  %.4f ms tanmaño %ix%i\n", elapsed, N, N);

    printf("Multiplicacion de matrices %s\n", error ? "incorrescta" : "correcta");

    // printf("Resultado de A * B GPU:\n");
    // printMatrix(h_C, N, N);
    // printf("Resultado de A * B CPU:\n");
    // printMatrix(h_C_cpu, N, N);

    // Liberar recursos
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
