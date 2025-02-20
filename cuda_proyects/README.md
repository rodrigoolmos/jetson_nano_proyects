# [00_vec_add](00_vec_add)
Evaluar el impacto del tamaño de blockSize (es decir, el número de threads por bloque) en el rendimiento de la suma de vectores en CUDA.

# [01_reg_vs_no_reg](01_reg_vs_no_reg)
Evaluar el efecto del uso de registros para almacenar variables que se utilizan repetidamente, comparándolo con el acceso directo a memoria global.

# [02_mac](02_mac)
Implementar una operación de MAC (Multiplicación Acumulativa) en CUDA utilizando mecanismos de sincronización entre hilos. Se estudia la sincronización dentro de un mismo bloque y se analiza el procesamiento en partes cuando el tamaño de los datos excede el número de hilos disponibles.

# [03_zero_copy](03_zero_copy)
Evaluar el efecto del uso de zero copy.

# [03_zero_copy](04_matmul)
Test diferentes formas de realizar multiplicacion de matrices.

