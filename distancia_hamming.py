def distancia_hamming(punto_a, punto_b):
    if len(punto_a) != len(punto_b):
        raise ValueError("Los puntos deben tener la misma cantidad de caracteristicas")

    diferencias = 0

    for i in range(len(punto_a)):
        if punto_a[i] != punto_b[i]:
            diferencias += 1

    return diferencias
