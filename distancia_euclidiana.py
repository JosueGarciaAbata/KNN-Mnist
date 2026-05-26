import numpy as np


def distancias_euclidianas(X_test, X_train, normas_entrenamiento=None):
    if normas_entrenamiento is None:
        normas_entrenamiento = np.sum(X_train * X_train, axis=1)

    
    normas_prueba = np.sum(X_test * X_test, axis=1)[:, None]
    distancias = normas_prueba + normas_entrenamiento[None, :] - 2.0 * (X_test @ X_train.T)
    np.maximum(distancias, 0.0, out=distancias)
    np.sqrt(distancias, out=distancias)
    return distancias
