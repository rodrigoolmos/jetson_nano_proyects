#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda_runtime.h>

#define N (1024 * 2)
// Coalescing Factor
#define COARSE_FACTOR 8
// Tiles of A
#define tiles_A_rows 64
#define tiles_A_cols 8

// Tiles of B
#define tiles_B_cols 64

__global__ void matmul_gpu(float *d_A, float *d_B, float *d_C)
{
    // Details regarding this thread
    const int by = blockIdx.y;
    const int bx = blockIdx.x;

    const int tx = threadIdx.x;

    // 1D -> 2D while loading A
    const int A_view_ty = tx / tiles_A_cols;
    const int A_view_tx = tx % tiles_A_cols;

    // 1D -> 2D while loading B
    const int B_view_ty = tx / tiles_B_cols;
    const int B_view_tx = tx % tiles_B_cols;

    // Working on C[row,col]
    const int row = tiles_A_rows * by + COARSE_FACTOR * (tx / tiles_B_cols);
    const int col = tiles_B_cols * bx + (tx % tiles_B_cols);

    // Allocating shared memory
    __shared__ float sh_A[tiles_A_rows][tiles_A_cols];
    __shared__ float sh_B[tiles_A_cols][tiles_B_cols];

    // Phases
    const int phases = ceil((float)N / tiles_A_cols);

    // Parallel mat mul
    float value[COARSE_FACTOR] = {0.0f};
    for (int phase = 0; phase < phases; phase++)
    {
        // Load Tiles into shared memory
        if ((by * tiles_A_rows + A_view_ty < N) && ((phase * tiles_A_cols + A_view_tx) < N))
            sh_A[A_view_ty][A_view_tx] = d_A[(by * tiles_A_rows + A_view_ty) * N + (phase * tiles_A_cols + A_view_tx)];
        else
            sh_A[A_view_ty][A_view_tx] = 0.0f;

        if (((phase * tiles_A_cols + B_view_ty) < N) && (bx * tiles_B_cols + B_view_tx < N))
            sh_B[B_view_ty][B_view_tx] = d_B[(phase * tiles_A_cols + B_view_ty) * N + (bx * tiles_B_cols + B_view_tx)];
        else
            sh_B[B_view_ty][B_view_tx] = 0.0f;
        __syncthreads();

        for (int k = 0; k < tiles_A_cols; k++)
        {
            float B_val_register = sh_B[k][B_view_tx];
            // Dot product
            for (int c = 0; c < COARSE_FACTOR; c++)
                value[c] += sh_A[B_view_ty * COARSE_FACTOR + c][k] * B_val_register;
        }
        __syncthreads();
    }

    // Assigning calculated value
    for (int c = 0; c < COARSE_FACTOR; ++c)
    {
        if ((row + c < N) && (col < N))
            d_C[(row + c) * N + (col)] = 1 * value[c];
    }
}

void printMatrix(const float *mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

// Multiplicación de matrices en CPU (implementación sencilla)
void matmul_cpu(float *h_A, float *h_B, float *h_C_cpu)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                sum += h_A[i * N + k] * h_B[k * N + j];
            }
            h_C_cpu[i * N + j] = sum;
        }
    }
}

int main()
{
    struct timeval start, stop;
    double elapsed;
    int error = 0;

    size_t bytes = N * N * sizeof(float);

    // Alojar memoria en la CPU
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    float *h_C_cpu = (float *)malloc(bytes);

    // Inicializar matrices en la CPU
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = (rand() % 20 - 10) / 10.0f;
        h_B[i] = (rand() % 20 - 10) / 10.0f;
        h_C[i] = 0.0f;
        h_C_cpu[i] = 0.0f;
    }

    // Alojar memoria en la GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    
    // Definir el tamaño del bloque y la cuadrícula en 2D
    dim3 gridSize(ceil(N / (float)(tiles_B_cols)), ceil(N / (float)(tiles_A_rows)));
    dim3 blockSize(tiles_A_rows * tiles_B_cols / COARSE_FACTOR);
    
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
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (fabs(h_C[i * N + j] - h_C_cpu[i * N + j]) > 0.1f)
            {
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
