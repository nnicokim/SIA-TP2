# SIA-TP2

### ALGORITMO GENETICO: APROXIMACION DE IMAGENES CON TRIANGULOS

Este proyecto implementa un motor de Algoritmos Geneticos (AG) diseñado para replicar una imagen objetivo utilizando una cantidad limitada de triangulos superpuestos y semitransparentes. Es la solucion correspondiente al Trabajo Practico de Algoritmos Geneticos para la materia Sistemas de Inteligencia Artificial.

### REQUISITOS PREVIOS
Para ejecutar este proyecto, necesitas tener instalado Python 3.8 o superior. Ademas, el motor utiliza algunas librerias externas estandar para el manejo matematico, procesamiento de imagenes y graficacion.

Para instalar todas las dependencias necesarias ejecutar el siguiente comando en tu terminal:
```bash
pip install numpy Pillow matplotlib pyyaml
```
### CONFIGURACION (Input)
El programa no requiere parametros extensos por linea de comandos. En su lugar, toda la configuracion se centraliza en un archivo *config.yaml* que debe estar ubicado en la raiz del proyecto.

Asegurate de configurar la ruta de tu imagen de prueba y los hiperparametros antes de ejecutar. Ejemplo de la estructura de *config.yaml*:
```yaml
image:
    path: "test_image.png"  (Ruta a la imagen objetivo)
    resolution: [64, 64]    (Resolucion de trabajo, se recomienda 64x64)

algorithm:
    pop_size: 50            (Tamano de la poblacion)
    n_triangles: 50         (Cantidad de triangulos por individuo)
    max_generations: 1000   (Condicion de corte por tiempo)
    target_fitness: 0.95    (Condicion de corte por aptitud. Usar 0 para ignorar)

selection:
    method: "ruleta"      (Opciones: elite, ruleta, boltzmann, universal, ranking, deterministic_tournament, probabilistic_tournament)

crossover:
    method: "one_point"   (Opciones: one_point, two_point, uniform, annular)

mutation:
    method: "gene"        (Opciones: gene, uniform, nouniform)
    rate: 0.1             (Probabilidad de mutacion)

survival:
    method: "exclusive"   (Opciones: additive, exclusive)

output:
    dir: "results"          (Carpeta donde se guardaran las imagenes)
    save_step: 50           (Guardar imagen cada X generaciones)
```

###  COMO EJECUTAR EL PROGRAMA
El proyecto cuenta con dos modos de ejecucion principales:

- Motor Principal (main.py)
Este es el flujo principal del programa. Ejecuta el algoritmo genetico basandose enteramente en las configuraciones del *config.yaml* y exporta visualmente la evolucion del mejor individuo.
    - Comando: 
        ```bash
        python main.py
        ```

    - Salida: Mostrara por consola el progreso del fitness y guardara las imagenes intermedias .png del mejor individuo en la carpeta definida en output.dir (por defecto /results).

- Pruebas y Metricas (benchmark.py)
Este script esta hecho para evaluar y comparar el rendimiento de los distintos metodos implementados (Seleccion, Cruza, Mutacion y Supervivencia).
    - **Comando**:
        ```bash
        python benchmark.py
        ```
    - **Salida**: Aisla cada categoria de operadores y prueba todos sus metodos manteniendo el resto constante. Genera graficos de linea (Fitness vs. Generaciones) y los guarda en sus respectivas subcarpetas dentro del directorio **figs/**. 

### ESTRUCTURA DEL PROYECTO

- *main.py*: Punto de entrada principal del algoritmo.

- *benchmark.py*: Script generador de metricas y graficos comparativos.

- *Engine.py*: Contiene el ciclo evolutivo y el calculo de aptitud (Fitness por ECM).

- *Individual.py*: Define la representacion genetica y la capacidad de renderizarse a imagen.

- *config.yaml*: Archivo de configuracion central.

- **methods/**: Directorio que contiene las interfaces y logicas de los operadores.

    - *factory.py*: Patron Factory para instanciar metodos dinamicamente.

    - *selection.py*: Metodos de seleccion.

    - *crossover.py*: Metodos de cruza.

    - *mutation.py*: Metodos de mutacion.

    - *survival.py*: Metodos de reemplazo generacional.