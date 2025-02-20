cmake_minimum_required(VERSION 3.10)
project(cuda LANGUAGES C CXX CUDA)

# Especificar el archivo fuente
add_executable(vec_add_slow vec_add_slow.cu)
add_executable(vec_add_fast vec_add_fast.cu)
add_executable(vec_add_test vec_add_test.cu)



# Configurar CUDA para que se compile correctamente
set_target_properties(vec_add_slow PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(vec_add_fast PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(vec_add_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Configurar la versión de CUDA, si es necesario
find_package(CUDA REQUIRED)
