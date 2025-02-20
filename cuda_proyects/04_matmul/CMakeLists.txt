cmake_minimum_required(VERSION 3.10)
project(cuda LANGUAGES C CXX CUDA)

# Buscar CUDA y cuBLAS
find_package(CUDAToolkit REQUIRED)

# Lista de ejecutables y sus archivos fuente
set(EXECUTABLES
    mat_mul_nvidia_cuda_core
    00_mat_mul
    01_mat_mul
    02_mat_mul
    03_mat_mul
    04_mat_mul
    05_mat_mul
    06_mat_mul
    07_mat_mul
    08_mat_mul
)

# Agregar cada ejecutable y configurar sus propiedades de CUDA
foreach(EXE IN LISTS EXECUTABLES)
    add_executable(${EXE} ${EXE}.cu)
    set_target_properties(${EXE} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 11  
    )
endforeach()

# Vincular CUDA y cuBLAS
target_link_libraries(mat_mul_nvidia_cuda_core PRIVATE CUDA::cublas CUDA::cudart)

# Imprimir mensaje de éxito
message(STATUS "Configuración CUDA lista")
