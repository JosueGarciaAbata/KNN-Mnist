# Reconocimiento de dígitos escritos a mano

## El problema que origina este proyecto

Leer un número escrito por una persona es algo que el ser humano hace sin esfuerzo, pero que para una computadora no es nada evidente. Un mismo dígito puede escribirse de mil formas distintas: con trazos gruesos o delgados, inclinados, torcidos o incompletos. No existe una regla fija que diga "esto es un cinco y esto es un seis", porque cada quien escribe diferente. Programar a mano todas esas variantes sería imposible.

Este proyecto nace justamente de esa dificultad. La pregunta de fondo es sencilla de enunciar y difícil de resolver: **¿cómo puede un programa aprender a reconocer números escritos a mano sin que nadie le dicte una sola regla explícita?** La respuesta está en dejar que el programa aprenda de ejemplos en lugar de instrucciones. Si le mostramos miles de imágenes de dígitos junto con su valor correcto, el programa puede descubrir por sí mismo los patrones que distinguen un número de otro.

Para resolverlo se usa el dataset **MNIST**, un conjunto clásico de 70.000 imágenes de dígitos manuscritos del 0 al 9, cada una de 28×28 píxeles en escala de grises. Es el punto de partida tradicional para entender cómo una máquina aprende a "ver".

El proyecto ataca el mismo problema desde **dos enfoques distintos**, lo que permite compararlos y entender sus diferencias de raíz:

1. Un clasificador **K-Nearest Neighbors (KNN)** implementado desde cero.
2. Una **red neuronal** construida con TensorFlow/Keras.

## La idea detrás de cada enfoque

### KNN: clasificar por parecido

El primer enfoque parte de una intuición muy humana: **una imagen probablemente representa el mismo número que las imágenes a las que más se parece.** Eso es exactamente lo que hace KNN. No construye ninguna fórmula ni ajusta parámetros internos; simplemente guarda en memoria todas las imágenes de entrenamiento con su etiqueta correcta.

Cuando llega una imagen nueva que debe clasificar, el algoritmo mide qué tan distinta es esa imagen frente a todas las que ya conoce, se queda con las `k` más cercanas (las más parecidas) y deja que voten: el número que más se repita entre esos vecinos es la respuesta. En este proyecto `k` vale 3, así que cada predicción la deciden los tres ejemplos más similares.

Lo interesante es **cómo se mide ese "parecido"**, porque no hay una única manera. El proyecto incluye tres formas de calcular la distancia entre dos imágenes, lo que permite experimentar y ver cuál funciona mejor:

- **Distancia euclidiana**: la distancia "en línea recta", la diferencia geométrica directa entre los píxeles de dos imágenes. Es la que se usa por defecto.
- **Distancia manhattan**: suma las diferencias absolutas píxel a píxel, como quien cuenta cuadras en una ciudad en cuadrícula en vez de cortar en diagonal.
- **Distancia hamming**: cuenta en qué proporción de píxeles las dos imágenes simplemente difieren.

KNN tiene una característica peculiar: **no tiene una fase real de entrenamiento.** "Entrenar" aquí solo significa memorizar los datos. Todo el trabajo ocurre en el momento de predecir, cuando hay que comparar la imagen nueva contra el conjunto completo. Por eso el cálculo se hace por lotes (`batch_size`), para no agotar la memoria comparando todo de golpe.

### Red neuronal: aprender los patrones

El segundo enfoque cambia por completo la filosofía. En lugar de comparar contra ejemplos guardados, una **red neuronal aprende los patrones** de los dígitos durante el entrenamiento y, una vez aprendidos, ya no necesita los datos originales para predecir.

La red toma cada imagen de 28×28, la aplana en un vector de 784 valores y la hace pasar por varias capas de neuronas que, capa tras capa, van transformando esos píxeles en una representación cada vez más abstracta. La última capa entrega diez probabilidades, una por cada dígito posible, y se elige la más alta como respuesta. A lo largo de varias épocas de entrenamiento la red ajusta sus parámetros internos para equivocarse cada vez menos.

La diferencia de fondo entre ambos enfoques es clara: **KNN no aprende, recuerda; la red neuronal no recuerda, aprende.**

## Cómo está organizado el proyecto

El código está pensado para ser legible y modular, separando cada responsabilidad en su propio archivo:

- **`knn_algoritmo.py`** — El corazón del clasificador KNN. Contiene la clase `KNN`, que memoriza los datos, calcula distancias por lotes y decide cada predicción por votación de los vecinos más cercanos.
- **`distancia_euclidiana.py`**, **`distancia_manhattan.py`**, **`distancia_hamming.py`** — Las tres formas de medir el parecido entre imágenes. Están separadas para que el clasificador pueda usar cualquiera de ellas de forma intercambiable.
- **`visualizacion_knn.py`** — Todas las funciones de apoyo visual: comparación de métricas, matrices de confusión, predicciones individuales y proyecciones con PCA. Como las imágenes viven en un espacio de 784 dimensiones imposible de dibujar, PCA las reduce a 2 dimensiones para poder mostrar gráficamente cómo el modelo separa los dígitos y dónde se equivoca.
- **`KNN.ipynb`** — El cuaderno que cuenta toda la historia del enfoque KNN paso a paso: carga del dataset, visualización, normalización, predicción y evaluación.
- **`reuronal_network_mnist.py`** y **`redes_neuronales.ipynb`** — El enfoque de red neuronal, desde la carga del dataset hasta el entrenamiento, las gráficas de aprendizaje, la evaluación y el análisis de sobreajuste.
- **`requirements.txt`** — Las dependencias necesarias.

## El flujo de trabajo, paso a paso

Ambos enfoques siguen una misma lógica de principio a fin, que refleja cómo se aborda en la práctica un problema de clasificación:

1. **Cargar los datos.** Se obtienen las imágenes de MNIST y sus etiquetas, separadas en un conjunto de entrenamiento y uno de prueba.
2. **Explorar.** Se muestran algunos dígitos y se revisa cuántos ejemplos hay de cada número, para confirmar que el dataset está balanceado y no favorece a ninguna clase.
3. **Preprocesar.** Los píxeles, que van de 0 a 255, se normalizan al rango de 0 a 1, y cada imagen de 28×28 se convierte en un vector de 784 valores para poder operar con ella.
4. **Entrenar.** En KNN esto significa simplemente memorizar los datos; en la red neuronal significa ajustar sus parámetros a lo largo de varias épocas.
5. **Predecir.** Se clasifican las imágenes del conjunto de prueba, que el modelo nunca usó para aprender.
6. **Evaluar.** Se mide el acierto (`accuracy`), se construye la matriz de confusión para ver en qué números se confunde el modelo y se comparan los errores entre entrenamiento y prueba para detectar sobreajuste.

## Cómo ejecutarlo

Se recomienda usar un entorno virtual e instalar las dependencias:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Luego se pueden abrir los cuadernos para recorrer cada enfoque de forma interactiva:

```bash
jupyter notebook KNN.ipynb
# o bien
jupyter notebook redes_neuronales.ipynb
```

El cuaderno de KNN carga MNIST directamente desde TensorFlow, por lo que funciona sin archivos adicionales. El script de la red neuronal (`reuronal_network_mnist.py`) espera encontrar el dataset en formato IDX dentro de una carpeta `mnist_dataset`.

## Tecnologías utilizadas

- **NumPy** para todo el cálculo numérico, que es la base de la implementación de KNN desde cero.
- **TensorFlow / Keras** para la red neuronal y para cargar el dataset MNIST.
- **scikit-learn** para las métricas de evaluación, las matrices de confusión y la reducción de dimensionalidad con PCA.
- **pandas** y **matplotlib** para organizar resultados y graficarlos.
- **Jupyter** como entorno para presentar el análisis de forma narrada.
