def distancia_euclidiana(punto_a, punto_b):
    if len(punto_a) != len(punto_b):
        raise ValueError("Los puntos deben tener la misma cantidad de caracteristicas")

    suma = 0

    for i in range(len(punto_a)):
        diferencia = punto_a[i] - punto_b[i]
        suma += diferencia * diferencia

    return suma ** 0.5
