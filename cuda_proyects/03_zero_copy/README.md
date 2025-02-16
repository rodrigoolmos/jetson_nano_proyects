# USO DE zero coppy;

## Introducción
En este proyecto se evalua el uso de zero copy entre el host y el device para una jetson orin nano

## Conclusiones
Las pruebas realizadas demuestran que el uso de zero copy mejora significativamente ya que no se necesita mover datos del host al device.
EJ: Suma de dos vecotores de 2^27 elementos

| tipo                          | Tiempo (ms) |
|----------------------------------|------------|
| **Zero Copy**     | 405.3478     |
| **No zero Copy**  | 98.6909 |