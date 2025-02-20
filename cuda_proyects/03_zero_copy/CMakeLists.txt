cmake_minimum_required(VERSION 3.10)
project(cuda LANGUAGES C CXX CUDA)

# Especificar el archivo fuente
add_executable(vec_add vec_add.cu)
add_executable(vec_add_zero_copy vec_add_zero_copy.cu)

# Configurar CUDA para que se compile correctamente
set_target_properties(vec_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(vec_add_zero_copy PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Configurar la versión de CUDA, si es necesario
find_package(CUDA REQUIRED)
