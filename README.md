# Red neuronal MNIST

Proyecto de red neuronal para reconocer digitos escritos a mano del dataset MNIST usando TensorFlow/Keras.

## Crear entorno virtual

Desde la carpeta del proyecto, ejecutar:

```powershell
python -m venv env
```

Activar el entorno virtual:

```powershell
.\env\Scripts\Activate.ps1
```

## Instalar dependencias

Con el entorno virtual activado:

```powershell
pip install -r requirements.txt
```

## Ejecutar Jupyter Lab

Con el entorno virtual activado:

```powershell
jupyter lab
```

Luego abrir el archivo:

```text
reuronal_network_mnist.ipynb
```

## Ejecutar como archivo Python

Si se desea generar el archivo `.py` desde el notebook:

```powershell
jupyter nbconvert --to script reuronal_network_mnist.ipynb
```

Y ejecutarlo:

```powershell
python reuronal_network_mnist.py
```
