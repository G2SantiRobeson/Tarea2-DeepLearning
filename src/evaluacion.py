"""Una única máscara de observaciones para MAE, RMSE y todos los modelos."""
import numpy as np
import pandas as pd


def metricas(real, pred, observada):
    """Devuelve MAE, RMSE y cobertura sobre la misma máscara de observación.

    real y pred están en µg/m³ y tienen la misma forma que observada. MAE es
    promedio de |error| y RMSE raíz del promedio de error²: penaliza más los picos.
    Una predicción no finita es un error, no un motivo para omitir esa hora.
    """
    real, pred, observada = np.asarray(real), np.asarray(pred), np.asarray(observada, dtype=bool)
    assert real.shape == pred.shape == observada.shape
    assert np.isfinite(pred).all(), 'No se pueden omitir fallos de predicción'
    assert np.array_equal(observada, np.isfinite(real))
    n = int(observada.sum())
    error = pred[observada] - real[observada]
    return dict(n_observados=n, n_total=real.size, cobertura=n / real.size,
                MAE=float(np.mean(np.abs(error))) if n else np.nan,
                RMSE=float(np.sqrt(np.mean(error ** 2))) if n else np.nan,
                negativos_pred=int((pred < 0).sum()))


def tabla_predicciones(df, objetivo, estacion, modelo, variante, semilla, bloque, js, pred):
    """Asocia (días,24) predicciones con origen, horizonte y observación original.

    Devuelve una fila por hora objetivo, conservando los faltantes como NaN y
    su bandera de presencia para calcular después métricas comparables.
    """
    assert pred.shape == (len(js), 24)
    indices = (js[:, None] + np.arange(24)).ravel()
    return pd.DataFrame(dict(estacion=estacion, objetivo=objetivo, modelo=modelo,
        variante=variante, ejecucion=str(semilla), bloque=bloque,
        origen=np.repeat(df.index[js] - pd.Timedelta(hours=1), 24),
        hora_objetivo=df.index[indices], horizonte=np.tile(np.arange(1, 25), len(js)),
        observacion=df[objetivo].to_numpy()[indices],
        observada=df[objetivo + '_observed'].to_numpy()[indices], estimacion=pred.ravel()))


def resumir(tabla):
    """Devuelve tablas de métricas globales y por horizonte para cada ejecución.

    Usa las filas horarias, sin promediar errores diarios con distinta cobertura.
    Las semillas se conservan separadas para mostrar su variabilidad.
    """
    columnas_grupo = ['estacion', 'objetivo', 'modelo', 'variante', 'ejecucion', 'bloque']
    globales, horizontes = [], []
    for valores, grupo in tabla.groupby(columnas_grupo, sort=False):
        info = dict(zip(columnas_grupo, valores))
        globales.append(info | metricas(grupo.observacion, grupo.estimacion, grupo.observada))
        for h, g in grupo.groupby('horizonte'):
            horizontes.append(info | dict(horizonte=h) | metricas(g.observacion, g.estimacion, g.observada))
    return pd.DataFrame(globales), pd.DataFrame(horizontes)


def verificar_comparabilidad(tabla):
    """Comprueba igualdad de horas y observaciones entre modelos de cada bloque.

    No devuelve otra tabla: detiene la evaluación si hay duplicados o si una
    ejecución omitió horas que las demás sí estaban obligadas a pronosticar.
    """
    columnas_caso = ['estacion', 'objetivo', 'bloque']
    for _, caso in tabla.groupby(columnas_caso):
        referencia = None
        for _, grupo in caso.groupby(['modelo', 'variante', 'ejecucion']):
            assert not grupo.hora_objetivo.duplicated().any()
            actual = grupo.sort_values('hora_objetivo')[['hora_objetivo', 'observacion', 'observada']].reset_index(drop=True)
            if referencia is None:
                referencia = actual
            else:
                assert referencia.equals(actual), 'Los modelos deben evaluar las mismas horas y observaciones'
