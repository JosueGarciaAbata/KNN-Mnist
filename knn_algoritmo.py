from distancia_euclidiana import distancia_euclidiana

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def entrenar(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predecir(self, x):
        distancias = []
        for i in range(len(self.X_train)):
            distancia = distancia_euclidiana(x, self.X_train[i])
            etiqueta = self.y_train[i]
            distancias.append((distancia, etiqueta))

        vecinos = self._obtener_k_vecinos(distancias)
        return self._votar(vecinos)

    def predecir_en_lote(self, X_test):
        predicciones = []

        for x in X_test:
            prediccion = self.predecir(x)
            predicciones.append(prediccion)

        return predicciones

    def _obtener_k_vecinos(self, distancias):
        distancias_ordenadas = sorted(distancias, key=self._obtener_distancia)
        return distancias_ordenadas[:self.k]

    def _obtener_distancia(self, distancia_y_etiqueta):
        # (distancia, etiqueta).
        return distancia_y_etiqueta[0]

    def _votar(self, vecinos):
        votos = {}

        for _, etiqueta in vecinos:
            if etiqueta not in votos:
                votos[etiqueta] = 0

            votos[etiqueta] += 1

        mejor_etiqueta = None
        mayor_cantidad = -1

        # votos {7: 2, 3: 1}
        for etiqueta in votos:
            if votos[etiqueta] > mayor_cantidad:
                mayor_cantidad = votos[etiqueta]
                mejor_etiqueta = etiqueta

        return mejor_etiqueta
