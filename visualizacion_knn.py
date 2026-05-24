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
        reporte_df = pd.DataFrame(reporte).T.round(3)
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

        ax.set_title(f"PCA - {nombre}")
        ax.set_xlabel("Componente principal 1")
        ax.set_ylabel("Componente principal 2")
        ax.legend()

    fig.colorbar(scatter, ax=axes, label="Prediccion KNN")
    plt.show()


def _crear_columnas(contenidos_html):
    cantidad_columnas = len(contenidos_html)

    return (
        f"<div style='display:grid; grid-template-columns:repeat({cantidad_columnas}, 1fr); "
        "gap:24px; align-items:start;'>"
        + "".join(f"<div>{contenido}</div>" for contenido in contenidos_html)
        + "</div>"
    )
