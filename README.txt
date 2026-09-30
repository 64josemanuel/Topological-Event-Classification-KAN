Clasificación Topológica de Eventos en Sistemas de Potencia mediante Redes KAN y GSP

Este repositorio contiene el código fuente para el entrenamiento, validación e interpretabilidad simbólica de un modelo de diagnóstico de fallas eléctricas. El sistema utiliza Procesamiento de Señales en Grafos (GSP) y Redes Neuronales de Kolmogorov-Arnold (KAN) para detectar eventos y extraer ecuaciones algebraicas a partir de simulaciones dinámicas.

Estructura del Repositorio

El sistema requiere y generará la siguiente estructura de directorios:

├── data/

│   ├── generators/          # Archivos .mat correspondientes a fallas de generador
│   ├── lines/               # Archivos .mat correspondientes a fallas de línea
│   └── loads/               # Archivos .mat correspondientes a cambios de carga
├── checkpoints_tesis/       # (Autogenerado después de la ejecución del código) Almacena tensores, modelos .pt y ecuaciones extraídas (.txt)
├── figuras_tesis/           # (Autogenerado después de la ejecución del código) Almacena gráficas de resultados en formato PDF
├── kan_master.py            # Script principal (Preprocesamiento, Entrenamiento KAN y Extracción Simbólica)

Requisitos de Datos (PST - MATLAB)

Los datos de entrada deben generarse mediante el Power System Toolbox (PST) de MATLAB.

Formato: Archivos .mat.

Contenido: Estructura denominada "sstr" con las variables "bus_v" (magnitud de voltaje en p.u.), "bus_freq" (frecuencia) y "t" (vector de tiempo).

Topología: El script extrae la matriz de admitancia leyendo las variables "bus" y "line" del primer archivo procesado.

Asignación dinámica de clases: El algoritmo escanea el directorio raíz de datos y asigna una clase independiente a cada subcarpeta que contenga archivos .mat. El tamaño de la capa de salida de la red KAN se ajusta automáticamente a la cantidad de subcarpetas detectadas.

Parámetros Modificables (kan_master.py)

Para adaptar el código a otros set de datos, modifique las siguientes variables en kan_master.py:

Directorio de datos (Si el repositorio cumple la estructura de directorios ocupar esta DIR_DATOS, caso contrario colocar la ruta donde se encuentran las carpetas con las clases.):
DIR_DATOS = r"."

Parámetros físicos:
frecuencia_base = 60.0 (Ajustar a 50.0 según el estándar de red).
limite_desviacion = 2.0 (Umbral de exclusión para nodos desconectados o inestables).

Hiperparámetros KAN:
En la declaración modelo_kan_mc = KAN(width=[dimension_entrada_mc, 8, num_clases_totales], grid=15, k=3):

width: Modificar el segundo valor (8) para alterar el número de neuronas ocultas.

grid: Modificar el número de intervalos de la cuadrícula de los splines (15).

Poda y Regresión Simbólica:
En la instrucción modelo_kan_mc.prune(node_th=1e-2, edge_th=1e-2):

Aumentar los umbrales (ej. 1e-1) para obtener ecuaciones más compactas.

Reducir los umbrales (ej. 1e-3) para conservar mayor complejidad y variables.

Instrucciones de Ejecución

Paso 1: Preparación del entorno
Asegúrese de instalar las dependencias requeridas: torch, sympy, numpy, scipy, seaborn, networkx, scikit-learn y pygsp.

Paso 2: Organización de datos
Coloque sus archivos .mat estructurados en subcarpetas dentro del directorio definido en DIR_DATOS.

Paso 3: Entrenamiento y extracción
Ejecute el script maestro para procesar los grafos, entrenar el modelo y generar el archivo ecuaciones_frontera.txt.

python kan_master.py

