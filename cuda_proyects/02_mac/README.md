# USO DE __syncthreads();

## Introducción
En este proyecto se emplean mecanismos de sincronización para implementar una operación de MAC (Multiplicación Acumulativa) en CUDA. La función __syncthreads() se utiliza para sincronizar a todos los hilos de un mismo bloque, garantizando que todos hayan alcanzado un determinado punto del código antes de continuar, lo que es crucial en operaciones colaborativas dentro del bloque.

## Conclusiones
Las pruebas realizadas demuestran que el uso de elementos de sincronización, como __syncthreads(), es efectivo para coordinar hilos dentro del mismo bloque. Sin embargo, si el tamaño de los datos excede la cantidad de hilos disponibles en un bloque, es necesario procesarlos en partes, ya que la sincronización solo es posible entre hilos que se encuentren en el mismo bloque.
