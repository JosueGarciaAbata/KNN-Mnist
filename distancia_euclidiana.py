import numpy as np


def distancias_euclidianas(X_test, X_train):
    distancias = []

    for a in X_test:
        distancia_a = np.sqrt(np.sum((X_train - a) ** 2, axis=1))
        distancias.append(distancia_a)

    return np.vstack(distancias)
