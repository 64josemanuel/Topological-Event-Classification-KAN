CLASIFICACION TOPOLOGICA DE EVENTOS EN SISTEMAS DE POTENCIA (GSP Y KAN)

Este repositorio contiene el codigo fuente para un modelo de diagnostico de fallas electricas. Utiliza Procesamiento de Senales en Grafos (GSP) y Redes Neuronales de Kolmogorov-Arnold (KAN) para detectar eventos y extraer ecuaciones algebraicas a partir de simulaciones dinamicas.

GUIA 
PASO 1: INSTALACION DE DEPENDENCIAS
Abre tu terminal e instala todas las librerias necesarias.  Ejecuta la siguiente linea de texto en tu consola o elige las que hagan falta:

pip install numpy scipy sympy torch seaborn matplotlib networkx tqdm scikit-learn PyGSP pykan

PASO 2: PREPARACION DE LOS DATOS
Tus datos deben ser generados mediante Power System Toolbox (PST) en MATLAB y exportados en formato .mat, o estar en el mismo formato.
Requisitos internos de los eventos: Debe existir una estructura llamada "sstr" que contenga las variables "bus_v" (voltaje), "bus_freq" (frecuencia) y "t" (tiempo).
Archivo de Topologia Base: Ademas de los eventos, el sistema necesita conocer la estructura de tu red. Debes tener un archivo .mat principal que contenga las matrices "bus" (informacion de generadores y cargas) y "line" (impedancias de las lineas). 
Como organizar los archivos: Crea una carpeta principal. Dentro de ella, coloca tu archivo de topologia base. Luego, crea subcarpetas para cada tipo de evento (ejemplo: "generators", "lines", "loads") y coloca los archivos de simulacion dentro. El sistema le asignara una clase a cada subcarpeta de manera automatica.

PASO 3: CONFIGURACION DE RED Y MATRICES
Si necesitas adaptar el modelo a tu propio sistema, abre el archivo kan_master.py y ajusta lo siguiente en las primeras lineas:

Uso de Topologia Fisica (USAR_YBUS_FISICA): Esta variable controla como se construye el grafo del sistema. Cambiala a True si deseas que el algoritmo lea los datos fisicos de tus lineas de transmision y construya la matriz de admitancia real. Dejala en False si no cuentas con esos datos o si prefieres que la Inteligencia Artificial deduzca las conexiones matematicamente (Graphical Lasso) basandose unicamente en el comportamiento de las oscilaciones.

Archivo de red base: Busca la linea de codigo que dice "90bus_wrew.mat". Este es el archivo predeterminado que el programa intenta leer para extraer la topologia ("bus" y "line"). Borra ese nombre y escribe exactamente el nombre del archivo .mat de tu propio sistema electrico.

Directorio: Modifica la variable DIR_DATOS para que tenga la ruta exacta hacia tu carpeta principal de datos. Si todo esta en el mismo lugar, dejalo como r"."

Parametros fisicos: Ajusta "frecuencia_base" a 50.0 o 60.0. Modifica "limite_desviacion" (por defecto 2.0) si necesitas cambiar el umbral para excluir nodos inestables.

Hiperparametros de la red KAN: Busca la definicion de width. Cambia el numero central (por defecto 8) para alterar las neuronas ocultas. Cambia grid=15 para modificar los intervalos de los splines.

Complejidad de las ecuaciones: Busca la instruccion prune. Si subes los umbrales (ejemplo 1e-1) obtendras ecuaciones mas compactas y simples. Si los bajas (ejemplo 1e-3) conservaras mas variables y mayor complejidad.

PASO 4: EJECUCION DEL PROGRAMA
Abre tu terminal en la ubicacion exacta donde guardaste el script principal y ejecuta la siguiente instruccion:

python kan_master.py

QUE OBTENDRAS AL FINALIZAR
El programa trabajara solo y creara automaticamente dos nuevas carpetas en tu equipo:

checkpoints_tesis: Aqui encontraras los tensores, los modelos entrenados guardados (.pt) y un archivo de texto llamado ecuaciones_frontera.txt con las formulas extraidas.

figuras_tesis: Aqui se guardaran todas las graficas de resultados y validacion en formato PDF.