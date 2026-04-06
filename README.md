# SIA-TP2

### ALGORITMO GENETICO: APROXIMACION DE IMAGENES CON TRIANGULOS

Este proyecto implementa un motor de Algoritmos Genéticos (AG) diseñado para replicar una imagen objetivo utilizando una cantidad limitada de triángulos superpuestos y semitransparentes. Es la solución correspondiente al Trabajo Práctico de Algoritmos Geneticos para la materia Sistemas de Inteligencia Artificial.

### REQUISITOS PREVIOS

Para ejecutar este proyecto, se necesita tener instalado Python 3.8 o superior. Además, el motor utiliza algunas librerias externas estandar para el manejo matemático, procesamiento de imágenes y graficación.

Para instalar todas las dependencias necesarias ejecutar el siguiente comando en tu terminal:

```bash
pip install numpy Pillow matplotlib pyyaml
```

### CONFIGURACION (Input)

El programa no requiere parametros extensos por linea de comandos. En su lugar, toda la configuración se centraliza en un archivo _config.yaml_ que debe estar ubicado en la raíz del proyecto.

Hay que segurarse de configurar la ruta de tu imagen de prueba y los hiperparametros antes de ejecutar. Ejemplo de la estructura de _config.yaml_:

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

### COMO EJECUTAR EL PROGRAMA

El proyecto cuenta con dos modos de ejecución principales:

- Motor Principal (main.py)
  Este es el flujo principal del programa. Ejecuta el algoritmo genético basandose enteramente en las configuraciones del _config.yaml_ y exporta visualmente la evolución del mejor individuo. - Comando:
  `bash
        python main.py
        `

      - Salida: Mostrará por consola el progreso del fitness y guardara las imagenes intermedias .png del mejor individuo en la carpeta definida en output.dir (por defecto /results).

- Pruebas y Metricas (benchmark.py)
  Este script esta hecho para evaluar y comparar el rendimiento de los distintos métodos implementados (Selección, Cruza, Mutacion y Supervivencia). - **Comando**:
  `bash
        python benchmark.py
        ` - **Salida**: Aisla cada categoria de operadores y prueba todos sus metodos manteniendo el resto constante. Genera graficos de linea (Fitness vs. Generaciones) y los guarda en sus respectivas subcarpetas dentro del directorio **figs/**.

### ESTRUCTURA DEL PROYECTO

- _main.py_: Punto de entrada principal del algoritmo.

- _benchmark.py_: Script generador de metricas y gráficos comparativos.

- _Engine.py_: Contiene el ciclo evolutivo y el calculo de aptitud (Fitness por ECM).

- _Individual.py_: Define la representación genética y la capacidad de renderizarse a imagen.

- _config.yaml_: Archivo de configuración central.

- **methods/**: Directorio que contiene las interfaces y logicas de los operadores.
  - _factory.py_: Patron Factory para instanciar métodos dinamicamente.

  - _selection.py_: Métodos de selección.

  - _crossover.py_: Métodos de cruza.

  - _mutation.py_: Métodos de mutación.

  - _survival.py_: Métodos de reemplazo generacional.
