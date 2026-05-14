#!/usr/bin/env python
# coding: utf-8

# # Red neuronal para reconocimiento de nÃºmeros
# 

# # Importacion de librerias

# In[44]:


import struct
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# ## Carga del dataset
# 
# El dataset MNIST esta guardado en la carpeta `mnist_dataset`. Como los archivos estan en formato IDX, se crean dos funciones para leer las imagenes y las etiquetas.
# 

# In[45]:


carpeta = Path("mnist_dataset")

def leer_imagenes(ruta):
    with open(ruta, "rb") as f:
        magic, n, filas, columnas = struct.unpack(">IIII", f.read(16))
        datos = np.frombuffer(f.read(), dtype=np.uint8)
    return datos.reshape(n, filas, columnas)

def leer_etiquetas(ruta):
    with open(ruta, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        datos = np.frombuffer(f.read(), dtype=np.uint8)
    return datos


# In[46]:


x_train = leer_imagenes(carpeta / "train-images.idx3-ubyte")
y_train = leer_etiquetas(carpeta / "train-labels.idx1-ubyte")

x_test = leer_imagenes(carpeta / "t10k-images.idx3-ubyte")
y_test = leer_etiquetas(carpeta / "t10k-labels.idx1-ubyte")


# In[47]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[48]:


np.unique(y_train)


# ## Visualizacion de algunos datos
# 
# Antes de entrenar la red neuronal, se muestran algunas imagenes del dataset para ver como son los numeros escritos a mano.
# 

# In[49]:


plt.figure(figsize=(10, 5))

for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(y_train[i])
    plt.axis("off")

plt.tight_layout()
plt.show()


# ## Cantidad de datos por clase
# 
# Se revisa cuantas imagenes hay para cada numero. Para verificar que el dataset esta balanceado.
# 

# In[50]:


pd.Series(y_train).value_counts().sort_index()


# In[51]:


pd.Series(y_train).value_counts().sort_index().plot(kind="bar")
plt.title("Cantidad de datos por numero")
plt.xlabel("Numero")
plt.ylabel("Cantidad")
plt.show()


# ## Preprocesamiento de datos
# 
# Las imagenes tienen pixeles con valores entre 0 y 255. Para que la red neuronal trabaje mejor, se dividen esos valores para 255 y asi quedan en un rango entre 0 y 1.
# 

# In[52]:


x_train = x_train / 255.0
x_test = x_test / 255.0


# In[53]:


print(x_train.min())
print(x_train.max())
print(x_train.shape)
print(x_test.shape)


# ## Construccion del modelo
# 
# Se crea una red neuronal secuencial. Primero se usa `Flatten` para convertir cada imagen de 28x28 en un vector. Luego se agregan tres capas ocultas con activacion `relu`, y al final una capa de salida con 10 neuronas porque hay 10 posibles numeros.
# 

# In[54]:


model = keras.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(10, activation="softmax")
])



# In[55]:


model.summary()


# ## Compilacion del modelo
# 
# Se compila el modelo indicando el optimizador, la funcion de perdida y la metrica que se va a observar. Como se tienen 10 clases y las etiquetas estan como numeros enteros, se usa `sparse_categorical_crossentropy`.
# 

# In[56]:


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ## Entrenamiento
# 
# Se entrena el modelo con los datos de entrenamiento. Tambien se separa una parte de esos datos para validacion, asi se puede observar como se comporta la red con datos que no usa directamente para ajustar los pesos.
# 

# In[57]:


history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=2
)


# ## Graficas del entrenamiento
# 
# Se revisa como cambian la perdida y la precision durante las epocas de entrenamiento.
# 

# In[58]:


historial = pd.DataFrame(history.history)
historial


# In[59]:


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(historial["loss"], label="train")
plt.plot(historial["val_loss"], label="validacion")
plt.title("Loss")
plt.xlabel("Epoca")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(historial["accuracy"], label="train")
plt.plot(historial["val_accuracy"], label="validacion")
plt.title("Accuracy")
plt.xlabel("Epoca")
plt.legend()

plt.show()


# ## Evaluacion del modelo
# 
# Se evalua el modelo con los datos de entrenamiento y con los datos de prueba. Tambien se compara el error de entrenamiento con el error de prueba, porque si el error de prueba es mucho mayor, puede ser una seÃ±al de sobreajuste.
# 
# 

# In[60]:


train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

error_train = 1 - train_accuracy
error_test = 1 - test_accuracy
diferencia_error = error_test - error_train

print("Accuracy train:", train_accuracy)
print("Accuracy test:", test_accuracy)

print("Error train:", error_train)
print("Error test:", error_test)

print("Diferencia entre error test y error train:", diferencia_error)



# ## Predicciones
# 
# El modelo devuelve una probabilidad para cada una de las 10 clases. Para obtener la prediccion final se usa `argmax`, que selecciona la clase con mayor probabilidad.
# 

# In[61]:


train_output = model.predict(x_train)
test_output = model.predict(x_test)

y_train_predic = np.argmax(train_output, axis=1)
y_test_predic = np.argmax(test_output, axis=1)


# ## Matriz de confusion
# 
# La matriz de confusion permite observar en que numeros acierta o se equivoca el modelo. Los valores de la diagonal principal representan las predicciones correctas.

# In[62]:


matriz = confusion_matrix(y_test, y_test_predic)

ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=np.arange(10)
).plot(cmap="Blues")

plt.title("Matriz de confusion")
plt.show()


# ### Validacion del modelo
# 
# Se toma una imagen del conjunto de prueba para ver que numero predice el modelo y con que probabilidad asigna cada clase del 0 al 9.
# 

# In[66]:


numero = 5

indices = np.where(y_test == numero)[0]
indice = np.random.choice(indices)

probabilidades = test_output[indice]
prediccion = y_test_predic[indice]
real = y_test[indice]

plt.imshow(x_test[indice], cmap="gray")
plt.title(f"Real: {real} - Prediccion: {prediccion}")
plt.axis("off")
plt.show()


# In[67]:


tabla_probabilidades = pd.DataFrame({
    "numero": np.arange(10),
    "probabilidad": probabilidades
})

tabla_probabilidades


# In[68]:


plt.bar(tabla_probabilidades["numero"], tabla_probabilidades["probabilidad"])
plt.xticks(np.arange(10))
plt.title("Probabilidad por cada numero")
plt.xlabel("Numero")
plt.ylabel("Probabilidad")
plt.show()


# In[ ]:




