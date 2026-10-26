"""Un ajuste SARIMA por caso; actualización de estado diaria, sin reajustar pesos."""
import time
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.stats.diagnostic import acorr_ljungbox


def ajustar(df, objetivo, c, prep, ruta):
    """Ajusta SARIMA con Train y guarda el modelo; devuelve diagnóstico y residuos.

    La estandarización usa prep, calculada solo con Train. Los NaN se conservan
    para que el filtro de Kalman trate las observaciones faltantes del objetivo.
    """
    y = df.loc[df.index < c['inicio_val'], objetivo]
    z = (y.to_numpy() - prep['media'][objetivo]) / prep['escala'][objetivo]
    modelo = SARIMAX(z, order=tuple(c['sarima_order']),
                     seasonal_order=tuple(c['sarima_seasonal']), trend='c',
                     enforce_stationarity=True, enforce_invertibility=True)
    inicio = time.perf_counter()
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always')
        res = modelo.fit(disp=False, maxiter=c['sarima_maxiter'], low_memory=True)
    segundos = time.perf_counter() - inicio
    res.save(ruta)
    residuos = np.asarray(res.resid)
    # Excluir arranque y horas sin etiqueta para el diagnóstico descriptivo.
    residuos[:max(168, res.loglikelihood_burn)] = np.nan
    residuos[~np.isfinite(z)] = np.nan
    limpios = residuos[np.isfinite(residuos)]
    lb = acorr_ljungbox(limpios, lags=[24, 48], model_df=2, return_df=True)
    diagnostico = dict(segundos_ajuste=segundos,
        convergio=bool(res.mle_retvals.get('converged', False)),
        iteraciones=int(res.mle_retvals.get('iterations', 0)),
        aic=float(res.aic), bic=float(res.bic), parametros=dict(zip(res.param_names, res.params.tolist())),
        numero_parametros=len(res.params),
        advertencias=list(dict.fromkeys(str(w.message) for w in avisos)),
        ljung_box=lb.reset_index(names='lag').to_dict('records'),
        nota_ljung_box='Descriptivo: al omitir faltantes se comprimen los rezagos; no prueba formal con grilla completa.',
        media_residuo=float(np.mean(limpios)), std_residuo=float(np.std(limpios)))
    return diagnostico, pd.DataFrame({'datetime': y.index, 'residuo_normalizado': residuos})


def pronosticar(df, objetivo, js, prep, c, ruta):
    """Carga un ajuste y devuelve (días,24) en las unidades originales del objetivo.

    js identifica los días futuros. Para cada emisión se incorporan mediciones
    solo hasta j-1; forecast(24) no accede a valores reales del día pronosticado.
    Actualizar estados con extend no vuelve a estimar los parámetros SARIMA.
    """
    estado = SARIMAXResults.load(ruta)
    # El respaldo de ajuste usa poca memoria. Recuperar estados de Train para
    # extend requiere filtrar UNA vez con los mismos parámetros, sin optimización.
    estado = estado.model.filter(estado.params, conserve_memory=0)
    cursor = int(df.index.searchsorted(pd.Timestamp(c['inicio_val'])))
    y = (df[objetivo].to_numpy() - prep['media'][objetivo]) / prep['escala'][objetivo]
    salida = []
    for j in js:
        assert j >= cursor
        if j > cursor:
            # Al emitir en j-1 se incorpora solo lo conocido hasta j-1.
            # extend filtra con parámetros FIJOS y admite observaciones NaN.
            estado = estado.extend(y[cursor:j])
        salida.append(np.asarray(estado.forecast(24)))
        cursor = j
    return np.stack(salida) * prep['escala'][objetivo] + prep['media'][objetivo]
