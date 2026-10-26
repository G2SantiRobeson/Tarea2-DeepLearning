"""Repetición del perfil horario anterior: rezagos 24 y 168, sin ajuste."""
import numpy as np


def predecir_naive(df, objetivo, js, rezago):
    """Devuelve (días,24) copiando el perfil de 24 o 168 horas atrás.

    df contiene una estación y js sus días objetivo. Se rellena una fuente
    histórica faltante solo con pasado, como en la preparación de las LSTM.
    """
    assert rezago in (24, 168)
    historia = df[objetivo].ffill().fillna(0).to_numpy()
    # j es la medianoche objetivo. j+23-rezago <= j-1: todo es conocido.
    return np.stack([historia[j-rezago:j-rezago+24] for j in js])
