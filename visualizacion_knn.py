def mostrar_comparacion_metricas(y_real, predicciones_por_metrica):
    import numpy as np
    import pandas as pd
    from IPython.display import HTML, display
    from sklearn.metrics import accuracy_score, classification_report

    resumenes = []

    for nombre, y_pred in predicciones_por_metrica.items():
        reporte = classification_report(
            y_real,
            y_pred,
            labels=np.arange(10),
            zero_division=0,
            output_dict=True,
        )
        reporte_df = pd.DataFrame(reporte).T.loc[[str(i) for i in range(10)]].round(3)
        accuracy = accuracy_score(y_real, y_pred)

        html = f"<h3>{nombre}</h3>"
        html += f"<p><b>Accuracy:</b> {accuracy:.4f}</p>"
        html += reporte_df.to_html()
        resumenes.append(html)

    display(HTML(_crear_columnas(resumenes)))


def mostrar_comparacion_matrices_confusion(y_real, predicciones_por_metrica):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cantidad_metricas = len(predicciones_por_metrica)
    fig, axes = plt.subplots(1, cantidad_metricas, figsize=(7 * cantidad_metricas, 5))
    axes = np.atleast_1d(axes)

    for ax, (nombre, y_pred) in zip(axes, predicciones_por_metrica.items()):
        matriz = confusion_matrix(y_real, y_pred, labels=np.arange(10))
        ConfusionMatrixDisplay(
            confusion_matrix=matriz,
            display_labels=np.arange(10),
        ).plot(cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"Matriz de confusion - {nombre}")

    plt.tight_layout()
    plt.show()


def mostrar_comparacion_pca(X_pca, y_real, predicciones_por_metrica):
    import numpy as np
    import matplotlib.pyplot as plt

    y_real = np.array(y_real)
    cantidad_metricas = len(predicciones_por_metrica)
    fig, axes = plt.subplots(
        1,
        cantidad_metricas,
        figsize=(7 * cantidad_metricas, 5),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for ax, (nombre, y_pred) in zip(axes, predicciones_por_metrica.items()):
        y_pred = np.array(y_pred)
        errores = y_pred != y_real

        scatter = ax.scatter(
            X_pca[~errores, 0],
            X_pca[~errores, 1],
            c=y_pred[~errores],
            cmap="tab10",
            alpha=0.65,
            s=25,
            label="Correctas",
        )
        ax.scatter(
            X_pca[errores, 0],
            X_pca[errores, 1],
            c="red",
            marker="x",
            s=45,
            label="Incorrectas",
        )

        ax.set_title(f"Pca - {nombre}")
        ax.set_xlabel("Componente principal 1")
        ax.set_ylabel("Componente principal 2")
        ax.legend()

    fig.colorbar(scatter, ax=axes, label="Prediccion KNN")
    plt.show()


def mostrar_pca_lote(
    contexto,
    y_real,
    predicciones_por_metrica,
    nombre_modelo="Manhattan",
    k=3,
    cantidad_entrenamiento_voronoi=1000,
    resolucion=160,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier

    X_train_flat = contexto["X_train_flat"]
    X_test_flat = contexto["X_test_flat"]
    cantidad_entrenamiento = contexto["cantidad_entrenamiento"]
    cantidad_prueba = len(y_real)

    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_train_flat[:cantidad_entrenamiento])

    X_test_pca = pca.transform(X_test_flat[:cantidad_prueba])

    if contexto["y_train_knn"] is None:
        mostrar_comparacion_pca(X_test_pca, y_real, predicciones_por_metrica)
        return

    cantidad_voronoi = min(cantidad_entrenamiento_voronoi, cantidad_entrenamiento)
    X_voronoi_pca = pca.transform(X_train_flat[:cantidad_voronoi])
    y_voronoi = contexto["y_train_knn"][:cantidad_voronoi]

    clasificador_2d = KNeighborsClassifier(
        n_neighbors=k,
        metric="manhattan",
    )
    clasificador_2d.fit(X_voronoi_pca, y_voronoi)

    margen = 0.8
    x_min = X_voronoi_pca[:, 0].min() - margen
    x_max = X_voronoi_pca[:, 0].max() + margen
    y_min = X_voronoi_pca[:, 1].min() - margen
    y_max = X_voronoi_pca[:, 1].max() + margen

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolucion),
        np.linspace(y_min, y_max, resolucion),
    )
    puntos_grid = np.c_[xx.ravel(), yy.ravel()]
    predicciones_grid = clasificador_2d.predict(puntos_grid).reshape(xx.shape)

    y_real = np.array(y_real)
    cantidad_metricas = len(predicciones_por_metrica)
    fig, axes = plt.subplots(
        1,
        cantidad_metricas,
        figsize=(7 * cantidad_metricas, 5),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for ax, (nombre, y_pred) in zip(axes, predicciones_por_metrica.items()):
        y_pred = np.array(y_pred)
        errores = y_pred != y_real

        ax.contourf(
            xx,
            yy,
            predicciones_grid,
            levels=np.arange(-0.5, 10.5, 1),
            cmap="tab10",
            alpha=0.18,
        )
        scatter = ax.scatter(
            X_test_pca[~errores, 0],
            X_test_pca[~errores, 1],
            c=y_pred[~errores],
            cmap="tab10",
            alpha=0.65,
            s=25,
            label="Correctas",
        )
        ax.scatter(
            X_test_pca[errores, 0],
            X_test_pca[errores, 1],
            c="red",
            marker="x",
            s=45,
            label="Incorrectas",
        )

        ax.set_title(f"Celdas Pca - {nombre_modelo}")
        ax.set_xlabel("Componente principal 1")
        ax.set_ylabel("Componente principal 2")
        ax.legend()

    fig.colorbar(scatter, ax=axes, label="Prediccion KNN")
    plt.show()


def mostrar_prediccion_individual(
    modelo,
    X_test_knn,
    y_test_knn,
    X_test_imagenes,
    X_train_flat,
    X_test_flat,
    cantidad_entrenamiento,
    indice=0,
    nombre_modelo="KNN",
    cantidad_contexto_pca=500,
    y_train_knn=None,
    cantidad_entrenamiento_voronoi=1000,
    mostrar_celdas=True,
    k=3,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier

    observacion = X_test_knn[indice]
    etiqueta_real = y_test_knn[indice]
    prediccion = modelo.predecir(observacion)

    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_train_flat[:cantidad_entrenamiento])

    cantidad_contexto = min(len(X_test_knn), cantidad_contexto_pca)
    X_contexto_pca = pca.transform(X_test_flat[:cantidad_contexto])

    if indice < cantidad_contexto:
        observacion_pca = X_contexto_pca[indice]
    else:
        observacion_pca = pca.transform([X_test_flat[indice]])[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(X_test_imagenes[indice], cmap="gray")
    axes[0].set_title(f"Real: {etiqueta_real} | Prediccion: {prediccion}")
    axes[0].axis("off")

    if mostrar_celdas and y_train_knn is not None:
        cantidad_voronoi = min(cantidad_entrenamiento_voronoi, cantidad_entrenamiento)
        X_voronoi_pca = pca.transform(X_train_flat[:cantidad_voronoi])
        y_voronoi = y_train_knn[:cantidad_voronoi]

        clasificador_2d = KNeighborsClassifier(
            n_neighbors=k,
            metric="manhattan",
        )
        clasificador_2d.fit(X_voronoi_pca, y_voronoi)

        margen = 0.8
        x_min = X_voronoi_pca[:, 0].min() - margen
        x_max = X_voronoi_pca[:, 0].max() + margen
        y_min = X_voronoi_pca[:, 1].min() - margen
        y_max = X_voronoi_pca[:, 1].max() + margen

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 160),
            np.linspace(y_min, y_max, 160),
        )
        puntos_grid = np.c_[xx.ravel(), yy.ravel()]
        predicciones_grid = clasificador_2d.predict(puntos_grid).reshape(xx.shape)

        axes[1].contourf(
            xx,
            yy,
            predicciones_grid,
            levels=np.arange(-0.5, 10.5, 1),
            cmap="tab10",
            alpha=0.18,
        )

    scatter = axes[1].scatter(
        X_contexto_pca[:, 0],
        X_contexto_pca[:, 1],
        c=y_test_knn[:cantidad_contexto],
        cmap="tab10",
        alpha=0.55,
        s=24,
        label="Digitos del lote",
    )
    axes[1].scatter(
        observacion_pca[0],
        observacion_pca[1],
        c="black",
        s=160,
        marker="*",
        label="Observacion individual",
    )
    axes[1].set_title(f"Individuo en celdas Pca - {nombre_modelo}")
    axes[1].set_xlabel("Componente principal 1")
    axes[1].set_ylabel("Componente principal 2")
    axes[1].legend()
    fig.colorbar(scatter, ax=axes[1], label="Etiqueta real")

    plt.tight_layout()
    plt.show()

    print("Etiqueta real:", etiqueta_real)
    print(f"Prediccion {nombre_modelo}:", prediccion)

    return prediccion


def crear_contexto_visualizacion(
    X_train_flat,
    X_test_flat,
    X_test_imagenes,
    X_test_knn,
    y_test_knn,
    cantidad_entrenamiento,
    y_train_knn=None,
):
    return {
        "X_train_flat": X_train_flat,
        "X_test_flat": X_test_flat,
        "X_test_imagenes": X_test_imagenes,
        "X_test_knn": X_test_knn,
        "y_test_knn": y_test_knn,
        "cantidad_entrenamiento": cantidad_entrenamiento,
        "y_train_knn": y_train_knn,
    }


def mostrar_individuo(modelo, contexto, indice=0, nombre_modelo="KNN"):
    return mostrar_prediccion_individual(
        modelo,
        contexto["X_test_knn"],
        contexto["y_test_knn"],
        contexto["X_test_imagenes"],
        contexto["X_train_flat"],
        contexto["X_test_flat"],
        contexto["cantidad_entrenamiento"],
        indice=indice,
        nombre_modelo=nombre_modelo,
        y_train_knn=contexto["y_train_knn"],
    )


def mostrar_celdas_voronoi_pca(
    contexto,
    nombre_modelo="Manhattan",
    k=3,
    cantidad_entrenamiento_voronoi=1000,
    resolucion=180,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier

    if contexto["y_train_knn"] is None:
        raise ValueError("El contexto necesita y_train_knn para dibujar las celdas.")

    cantidad = min(cantidad_entrenamiento_voronoi, contexto["cantidad_entrenamiento"])
    X_train = contexto["X_train_flat"][:cantidad]
    y_train = contexto["y_train_knn"][:cantidad]

    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train)

    clasificador_2d = KNeighborsClassifier(
        n_neighbors=k,
        metric="manhattan",
    )
    clasificador_2d.fit(X_train_pca, y_train)

    margen = 0.8
    x_min, x_max = X_train_pca[:, 0].min() - margen, X_train_pca[:, 0].max() + margen
    y_min, y_max = X_train_pca[:, 1].min() - margen, X_train_pca[:, 1].max() + margen

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolucion),
        np.linspace(y_min, y_max, resolucion),
    )
    puntos_grid = np.c_[xx.ravel(), yy.ravel()]
    predicciones_grid = clasificador_2d.predict(puntos_grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(
        xx,
        yy,
        predicciones_grid,
        levels=np.arange(-0.5, 10.5, 1),
        cmap="tab10",
        alpha=0.22,
    )
    scatter = plt.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        c=y_train,
        cmap="tab10",
        s=18,
        alpha=0.75,
        edgecolors="none",
    )

    plt.title(f"Celdas aproximadas en Pca - {nombre_modelo}")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.colorbar(scatter, label="Etiqueta real")
    plt.show()


def _crear_columnas(contenidos_html):
    cantidad_columnas = len(contenidos_html)

    return (
        f"<div style='display:grid; grid-template-columns:repeat({cantidad_columnas}, 1fr); "
        "gap:24px; align-items:start;'>"
        + "".join(f"<div>{contenido}</div>" for contenido in contenidos_html)
        + "</div>"
    )
