cmake_minimum_required(VERSION 3.10)
project(Nitefury LANGUAGES C)

# Especificar el archivo fuente
add_executable(axi_lite axi_lite.c)
add_executable(axi_full axi_full.c)