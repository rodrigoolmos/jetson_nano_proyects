cmake_minimum_required(VERSION 3.10)
project(cuda LANGUAGES C CXX CUDA)

# Especificar el archivo fuente
add_executable(mac mac.cu)
add_executable(mac_big mac_big.cu)

# Configurar CUDA para que se compile correctamente
set_target_properties(mac PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(mac_big PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Configurar la versión de CUDA, si es necesario
find_package(CUDA REQUIRED)
