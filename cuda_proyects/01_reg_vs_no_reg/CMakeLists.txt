cmake_minimum_required(VERSION 3.10)
project(cuda LANGUAGES C CXX CUDA)

# Especificar el archivo fuente
add_executable(reg_vs_no_reg reg_vs_no_reg.cu)

# Configurar CUDA para que se compile correctamente
set_target_properties(reg_vs_no_reg PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Configurar la versión de CUDA, si es necesario
find_package(CUDA REQUIRED)
