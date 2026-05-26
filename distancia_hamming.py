import numpy as np


def distancias_hamming(X_test, X_train, normas_entrenamiento=None):
    bloques = []

    for x in X_test:
        bloques.append(np.mean(X_train != x, axis=1))

    return np.vstack(bloques)
