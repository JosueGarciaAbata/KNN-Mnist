import numpy as np


class KNN:
    def __init__(self, k=3, batch_size=100):
        self.k = k
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None
        self._train_normas = None

    def entrenar(self, X_train, y_train):
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=np.int64)

        if self.X_train.ndim != 2:
            raise ValueError("X_train debe ser una matriz de datos")

        if len(self.X_train) != len(self.y_train):
            raise ValueError("X_train y y_train deben tener la misma cantidad de observaciones")

        self._train_normas = np.sum(self.X_train * self.X_train, axis=1)

    def predecir(self, x):
        return int(self.predecir_en_lote([x], batch_size=1)[0])

    def predecir_en_lote(self, X_test, batch_size=None):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Primero debe entrenar el modelo")

        X_test = np.asarray(X_test, dtype=np.float32)

        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        if X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError("Los datos de prueba deben tener la misma cantidad de caracteristicas")

        batch_size = batch_size or self.batch_size
        predicciones = np.empty(X_test.shape[0], dtype=np.int64)

        for inicio in range(0, X_test.shape[0], batch_size):
            fin = min(inicio + batch_size, X_test.shape[0])
            predicciones[inicio:fin] = self._predecir_lote_numpy(X_test[inicio:fin])

        return predicciones.tolist()

    def _predecir_lote_numpy(self, X_batch):
        test_normas = np.sum(X_batch * X_batch, axis=1)[:, None]
        distancias = test_normas + self._train_normas[None, :] - 2.0 * (X_batch @ self.X_train.T)
        np.maximum(distancias, 0.0, out=distancias)

        indices_vecinos = np.argpartition(distancias, self.k - 1, axis=1)[:, : self.k]
        etiquetas_vecinos = self.y_train[indices_vecinos]

        predicciones = np.empty(X_batch.shape[0], dtype=np.int64)
        for i, etiquetas in enumerate(etiquetas_vecinos):
            predicciones[i] = np.bincount(etiquetas, minlength=10).argmax()

        return predicciones
