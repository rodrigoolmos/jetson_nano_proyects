# Reporte de Rendimiento: Impacto del uso de registros en el Kernel CUDA

## Introducción

En este proyecto se evalúa el impacto de utilizar registros para almacenar una variable de uso frecuente en un kernel de CUDA. Al guardar temporalmente en un registro la variable que se utiliza múltiples veces, se reducen los accesos a la memoria global, lo que potencialmente mejora de forma significativa el rendimiento.

Además, se incluye la posibilidad de modificar el tamaño de bloque (blockSize) para observar cómo esta configuración puede influir en el tiempo de ejecución del kernel.

## Conclusiones

El uso de registros para almacenar variables que se utilizan reiteradamente en el kernel evita accesos frecuentes a la memoria global. Como consecuencia, se observa una reducción significativa en el tiempo de ejecución.
