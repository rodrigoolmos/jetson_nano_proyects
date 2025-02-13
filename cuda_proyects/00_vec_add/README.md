# Reporte de Rendimiento: Impacto del `blockSize` en el Kernel CUDA

## Introducción

En este proyecto se han realizado pruebas para analizar el impacto del parámetro `blockSize` en el rendimiento del kernel CUDA, específicamente en la suma de vectores de números complejos. El objetivo principal fue determinar la configuración óptima que maximice el aprovechamiento de los recursos de la GPU (SMs y CUDA cores) y, como consecuencia, minimizar los tiempos de ejecución.

## Conclusiones

Las pruebas realizadas demuestran que establecer `blockSize` al valor máximo permitido por la GPU (1024) resulta en el mejor rendimiento para el kernel evaluado. Esta configuración maximiza la ocupación de los SMs y el uso de los CUDA cores, lo que se traduce en una ejecución más rápida y eficiente. Estos hallazgos resaltan la importancia de ajustar adecuadamente los parámetros de ejecución según las características del hardware y del algoritmo, siendo el `blockSize` uno de los factores críticos en la optimización de aplicaciones CUDA.
