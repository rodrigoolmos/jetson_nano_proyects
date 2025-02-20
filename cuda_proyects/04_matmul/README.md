# Experimentando con MatMul

Este repositorio contiene diversas implementaciones de la multiplicación de matrices (MatMul) en CUDA, cada una con diferentes optimizaciones para mejorar el rendimiento. A continuación se detallan las distintas versiones y sus características principales:

## Implementaciones

### [00_mat_mul.cu: Implementación Básica](00_mat_mul.cu)
- Múltiples accesos a memoria sin utilizar un registro temporal para la acumulación, lo que genera accesos redundantes.
- No se logra coalescencia; los hilos no realizan accesos consecutivos a la memoria.

### [01_mat_mul.cu: Uso de Registro Temporal para Acumulación](01_mat_mul.cu)
- Se utiliza un registro para evitar múltiples accesos a memoria durante la acumulación.
- Aún no se consigue coalescencia, por lo que los accesos a la memoria siguen sin ser consecutivos.

### [02_mat_mul.cu: Acceso Consecutivo Coalesced + Registro](02_mat_mul.cu)
- Se emplea un registro para evitar múltiples accesos durante la acumulación.
- La modificación en la declaración de los hilos permite realizar accesos consecutivos (coalescencia) a la memoria.

### [03_mat_mul.cu: Tiled Matrix Multiplication](03_mat_mul.cu)
- Uso de registro temporal para la acumulación.
- Los hilos acceden de forma consecutiva a la memoria gracias a una organización optimizada.
- Se utilizan porciones de las matrices almacenadas en memoria compartida (__shared__ memory) para lograr accesos mucho más rápidos.

### [04_mat_mul.cu: Acceso Consecutivo Coalesced r/w + Registro](04_mat_mul.cu)
- Se utiliza un registro para evitar múltiples accesos a memoria al acumular.
- La optimización en la declaración de los hilos permite accesos consecutivos a la memoria.
- Se guarda un vector de sumas en memoria compartida, lo que permite copiar el resultado final de forma coalescente (accesos secuenciales).

### [05_mat_mul.cu: Acceso Consecutivo Coalesced r/w + Registro (Versión 5)](05_mat_mul.cu)
- Uso de registro temporal para evitar accesos redundantes durante la acumulación.
- Los hilos acceden de forma consecutiva a la memoria gracias a la reorganización de la declaración de hilos.
- Se utilizan porciones de las matrices en memoria compartida para mejorar la velocidad de acceso.
- Se emplean registros para reutilizar elementos de la matriz A.

### [06_mat_mul.cu: Acceso Consecutivo Coalesced r/w + Registro (Versión 6)](06_mat_mul.cu)
- Se utiliza un registro temporal para evitar múltiples accesos a memoria al acumular.
- La declaración optimizada de los hilos permite acceder a la memoria de forma consecutiva.
- Se emplea memoria compartida para almacenar porciones de las matrices, acelerando significativamente los accesos.
- Se usan registros para reutilizar elementos tanto de la matriz A como de la B.

### [07_mat_mul.cu: Acceso Consecutivo Coalesced r/w + Registro (Versión 7)](07_mat_mul.cu)
- Uso de registro temporal para la acumulación, reduciendo accesos redundantes.
- La organización de los hilos permite acceder a la memoria de manera consecutiva (coalescencia).
- Se utiliza memoria compartida para almacenar porciones de las matrices, lo que mejora notablemente la velocidad de acceso.
- Se reutilizan elementos de ambas matrices (A y B) mediante registros.
- Los datos se cargan de forma vectorial.

### [08_mat_mul.cu: Acceso Consecutivo Coalesced r/w + Registro (Versión 8)](08_mat_mul.cu)
- Se utiliza un registro temporal para evitar múltiples accesos a memoria durante la acumulación.
- La optimización en la declaración de los hilos permite accesos consecutivos a la memoria.
- Se emplea memoria compartida para almacenar porciones de las matrices, logrando accesos mucho más rápidos.
- Se reutilizan elementos de las matrices A y B mediante registros.
- Los datos se cargan de forma vectorial.
- Se implementa la técnica de zero copy para eliminar transferencias innecesarias entre host y device.


| Archivo         | ms                |
|-----------------|-------------------|
| 00_mat_mul.cu   | 7115.5250         |
| 01_mat_mul.cu   | 1868.0180         |
| 02_mat_mul.cu   | 215.2720          |
| 03_mat_mul.cu   | 205.8960          |
| 04_mat_mul.cu   | 208.8960          |
| 05_mat_mul.cu   | 113.8960          |
| 06_mat_mul.cu   | 92.9390           |
| 07_mat_mul.cu   | 77.7020           |
| 08_mat_mul.cu   | 67.7020           |
