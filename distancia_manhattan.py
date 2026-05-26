import numpy as np


def distancias_manhattan(X_test, X_train, normas_entrenamiento=None):
    bloques = []

    for x in X_test:
        bloques.append(np.sum(np.abs(X_train - x), axis=1))

    return np.vstack(bloques)
