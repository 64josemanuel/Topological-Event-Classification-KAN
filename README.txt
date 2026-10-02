CLASIFICACION TOPOLOGICA DE EVENTOS EN SISTEMAS DE POTENCIA (GSP Y KAN)

Este repositorio contiene el codigo fuente para un modelo de diagnostico de fallas electricas. Utiliza Procesamiento de Senales en Grafos (GSP) y Redes Neuronales de Kolmogorov-Arnold (KAN) para detectar eventos y extraer ecuaciones algebraicas a partir de simulaciones dinamicas.

GUIA PASO A PASO

PASO 1: INSTALACION DE DEPENDENCIAS
Abre tu terminal e instala todas las librerias necesarias. Ejecuta la siguiente linea de texto en tu consola o elige las que hagan falta:

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

El programa creara automaticamente dos carpetas ("checkpoints_tesis" y "figuras_tesis"). Estas carpetas sirven para no tener que recalcular todo desde cero si vuelves a ejecutar el codigo y para almacenar tus resultados visuales.

DIRECTORIO DE CHECKPOINTS (checkpoints_tesis)
Guarda el progreso del sistema en diferentes etapas (pasos). Si el programa se interrumpe, retomara desde el ultimo checkpoint guardado.

dataset_crudo_master.pkl: Alcanza el Paso 1. Guarda los datos extraidos de MATLAB, la matriz fisica y las mascaras de nodos para no volver a leer los archivos .mat.

checkpoint_paso2.pkl: Alcanza el Paso 2. Guarda las matrices de topologia inferidas matematicamente (Graphical Lasso) y la matriz hibrida final.

dataset_tensores_fisica.pt: Alcanza el Paso 3. Guarda los datos ya convertidos en tensores de PyTorch listos para entrenar el modelo binario (Normal vs Transitorio).

modelo_kan_binario.pt y historial_optimizacion_binario.pkl: Alcanzan el Paso 4. Guardan los pesos matematicos de la red neuronal binaria ya entrenada y su historial de error, evitando reentrenar.

dataset_tensores_mc_fisico.pt: Alcanza el Paso 5. Guarda los tensores divididos en 5 ventanas de tiempo para la clasificacion multiclase.

modelo_kan_multiclase.pt: Alcanza el Paso 6. Guarda los pesos de la red neuronal multiclase ya entrenada.

stats_simbolicas.pkl: Alcanza el Paso 7. Guarda las estadisticas de la regresion simbolica (cuantas variables sobrevivieron y que funciones matematicas se usaron).

ecuaciones_frontera.txt: Archivo de texto final con las formulas matematicas extraidas.

COMO LEER LAS ECUACIONES FRONTERA (ecuaciones_frontera.txt)
Este archivo contiene las formulas matematicas explicitas que la IA aprendio para clasificar cada evento. En lugar de ser una "caja negra", el modelo te entrega una ecuacion por cada clase.
Las variables en las ecuaciones se leen de la siguiente manera:

El simbolo "λ" (lambda) representa la magnitud de la Transformada de Fourier en Grafos (GFT).

"V", "f_gen" o "f_load" indican si la senal proviene del voltaje, la frecuencia de un generador o la frecuencia de una carga.

El numero que le sigue (ejemplo: 0, 5, 12) es el indice de frecuencia del grafo (el nodo ).

La terminacion "W1" a "W5" indica la ventana de tiempo donde se observo esa senal (W1=1-2s, W2=2-5s, W3=5-10s, W4=10-15s, W5=15-20s).
Ejemplo de lectura: Si ves "λV_3_W2", significa "La magnitud del espectro de voltaje en la frecuencia de grafo 3, observada durante la ventana de tiempo 2".

DIRECTORIO DE FIGURAS (figuras_tesis)
Contiene las graficas en formato PDF. Explicacion de cada una:

Figuras de Topologia (Paso 2)

Fig2_A_Optimization.pdf: Muestra como la IA encontro la topologia del sistema. Eje X: Valor de regularizacion Alpha (escala logaritmica). Eje Y: Numero de aristas inferidas.

Fig2_B_Matrices.pdf: Mapas de calor comparando la matriz fisica, la inferida matematicamente y la hibrida final. Los ejes X e Y representan los nodos del sistema.

Fig2_C_Spectrum.pdf: Espectro de la Transformada de Grafos (GFT). Eje X: Indice de frecuencia del grafo (k). Eje Y: Valor propio (Eigenvalue).

Fig2_D_Modes.pdf: Diagramas de red que muestran la matriz fisica y las oscilaciones globales sobre la topologia.

Figuras de Datos Espectrales (Paso 3)

Fig3_A_Spectral_Contrast.pdf: Compara las senales normales vs transitorias. Eje X: Indice de frecuencia del grafo (k). Eje Y: Magnitud absoluta |GFT|.

Fig3_B_Tensor_Anatomy.pdf: Muestra como se empaquetan los datos que entran a la IA. Eje X: Indice del tensor (tamano = 3 veces el numero de nodos). Eje Y: Magnitud. Muestra los espectros de Voltaje, Frecuencia de Generador y Frecuencia de Carga.

Fig3_C_PCA_Separability.pdf: Proyeccion 2D para ver si los datos son separables. Eje X e Y: Componentes principales 1 y 2 (porcentaje de varianza).

Figuras de Evaluacion Binaria KAN (Paso 4)

Fig4_A_KAN_Evaluation.pdf: Tres graficas de rendimiento. 1) Matriz de confusion (Prediccion vs Condicion Real en %). 2) Curva ROC (Eje X: Tasa de falsos positivos, Eje Y: Tasa de verdaderos positivos). 3) Distribucion de certeza (Eje X: Salida continua del modelo, Eje Y: Densidad de probabilidad).

Fig4_B_Optimization_Dynamics.pdf: Dinamica de entrenamiento. Izquierda (Convergencia): Eje X: Iteraciones, Eje Y: Error Cuadratico Medio. Derecha (Evolucion de parametros): Eje X: Iteraciones, Eje Y: Penalizacion por poda.

Fig4_D_Spline_Evolution.pdf: Muestra como se deformo la funcion de activacion. Eje X: Valor de entrada (x). Eje Y: Funcion aprendida.

Figuras de Multiclase (Pasos 5 y 6)

Fig5_A_Kinematic_Evolution.pdf: Mapa de calor de la evolucion del evento en el tiempo. Eje X: Ventanas de tiempo (W1 a W5). Eje Y: Frecuencia de grafo (k). Color: Magnitud |GFT| del voltaje.

Fig5_B_Multiclass_Balance.pdf: Grafica de barras con el balance de datos. Eje X: Clases de eventos. Eje Y: Muestras totales.

Fig6_A_Multiclass_ROC.pdf: Curvas ROC para multiples clases. Eje X: Falsos positivos. Eje Y: Verdaderos positivos.

Fig6_B_KAN_Interpretability.pdf: Interpretabilidad multiclase. 1) Importancia de variables (Eje X: Ranking de variables, Eje Y: Norma L1 de activacion). 2) Redistribucion de la cuadricula KAN. 3) Deformacion final del spline.

Figuras Simbolicas (Paso 7)

Fig7_Symbolic_Interpretation.pdf: Impacto de la regresion simbolica. 1) Cantidad de variables antes y despues de podar la red (Eje Y: Numero de variables). 2) Frecuencia de uso de primitivas (seno, exponente, al cuadrado) elegidas por la IA. 3) Mapa de calor que muestra en que ventana de tiempo (Eje X) y para que clase (Eje Y) se concentran las variables analiticas extraidas.