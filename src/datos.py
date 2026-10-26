"""Lectura, preparación causal y ventanas diarias.

En estas funciones, df es la tabla horaria de una estación, c es la configuración
y prep contiene columnas, medias y escalas aprendidas exclusivamente en Train.
"""
import numpy as np
import pandas as pd


def cargar(ruta, estacion, c):
    """Lee el CSV y devuelve una estación, con datetime como índice horario.

    Comprueba grilla, unicidad y máscaras antes de construir secuencias: así una
    ventana de 48 filas representa 48 horas y no oculta saltos en el calendario.
    """
    df = pd.read_csv(ruta, parse_dates=['datetime'])
    df = df.loc[df.site.eq(estacion)].set_index('datetime').sort_index()
    assert df.index.is_unique, 'Clave estación-hora duplicada'
    df = df.loc[(df.index >= c['inicio']) & (df.index < c['fin'])].copy()
    esperado = pd.date_range(c['inicio'], c['fin'], freq='h', inclusive='left')
    assert df.index.equals(esperado), 'La grilla debe ser horaria y completa'
    for col in [x.removesuffix('_observed') for x in df if x.endswith('_observed')]:
        assert df[col + '_observed'].eq(df[col].notna()).all(), col
    return df


def particiones(c):
    """Devuelve la tabla de bloques [inicio, fin), horas y porcentaje temporal.

    Los límites provienen de config.py y deben estar ordenados a medianoche,
    para asignar cada día objetivo completo a Train, Validation o Test.
    """
    limites = [c[k] for k in ['inicio', 'inicio_val', 'inicio_test', 'fin']]
    fechas = pd.to_datetime(limites)
    assert all(t.hour == 0 and t.minute == 0 for t in fechas)
    assert all(a < b for a, b in zip(fechas, fechas[1:]))
    total = (fechas[-1] - fechas[0]).total_seconds() / 3600
    filas = []
    for nombre, a, b in zip(['train', 'val', 'test'], fechas, fechas[1:]):
        horas = int((b - a).total_seconds() / 3600)
        filas.append(dict(bloque=nombre, inicio=a, fin_exclusivo=b,
                          horas=horas, porcentaje=100 * horas / total))
    return pd.DataFrame(filas)


def calendario(fechas):
    """Convierte fechas en seis señales cíclicas conocidas al emitir el pronóstico.

    Devuelve (horas, 6): seno/coseno de hora, día semanal y día anual. Esta
    representación mantiene próximas horas como 23:00 y 00:00 en el ciclo diario.
    """
    # Seis variables conocidas a priori; no son mediciones del futuro.
    fases = [np.asarray(fechas.hour) / 24,
             np.asarray(fechas.day_of_week) / 7,
             (np.asarray(fechas.day_of_year) - 1) / 365.25]
    return np.column_stack([f(2 * np.pi * a) for a in fases
                            for f in [np.sin, np.cos]]).astype('float32')


def ajustar_preparacion(df, objetivo, c):
    """Devuelve medias y escalas del objetivo y meteorología de Train observado.

    Reutilizar estos mismos valores en Validation y Test evita que información
    de los períodos futuros influya en la normalización del entrenamiento.
    """
    columnas = [objetivo] + c['meteorologia']
    train = df.loc[df.index < c['inicio_val'], columnas]
    assert train.notna().any().all(), 'Variable sin observaciones en Train'
    # Estadísticos de observaciones originales de Train, ddof=0 como StandardScaler.
    return dict(columnas=columnas, media=train.mean().to_dict(),
                escala=train.std(ddof=0).replace(0, 1).to_dict(),
                imputacion='ffill; cero constante si no existe pasado',
                ajuste_fin_exclusivo=c['inicio_val'])


def preparar(df, prep):
    """Devuelve valores normalizados, máscaras originales y calendario por hora.

    Solo las entradas históricas se rellenan con pasado disponible. Las máscaras
    distinguen una medición original de un reemplazo, incluido un cero imputado.
    """
    cols = prep['columnas']
    observadas = df[cols].notna().to_numpy(dtype='float32')
    # No interpolar ni bfill: un dato posterior no puede rellenar el pasado.
    historicas = df[cols].ffill().fillna(0)
    z = ((historicas - pd.Series(prep['media'])) /
         pd.Series(prep['escala'])).to_numpy(dtype='float32')
    return z, observadas, calendario(df.index)


def dias_objetivo(df, c, bloque):
    """Devuelve los índices de las medianoches objetivo del bloque solicitado.

    El índice j identifica el primer horizonte y j-1 el origen de las 23:00.
    La historia puede pertenecer al bloque anterior porque ya estaba disponible.
    """
    fila = particiones(c).set_index('bloque').loc[bloque]
    fechas = pd.date_range(fila.inicio, fila.fin_exclusivo, freq='D', inclusive='left')
    js = df.index.get_indexer(fechas)
    # Siete días iniciales como contexto común, suficientes para Naive semanal.
    js = js[js >= max(c['ventana'], 168)]
    assert len(js), 'Bloque sin días evaluables'
    assert np.all(js + c['horizonte'] <= len(df))
    return js


def ventanas(df, objetivo, variante, c, prep, bloque):
    """Construye un ejemplo diario y devuelve entradas, etiquetas y sus máscaras.

    x tiene forma (días, 48, F), con F=8 en base y F=18 en meteo. futuro contiene
    solo calendario (días, 24, 6); y y mask tienen forma (días, 24). js identifica
    los días objetivo para vincular después cada salida con su fecha real.
    """
    # j es el índice de las 00:00 del día objetivo; j-1 es el origen (23:00).
    # x: historia (días, ventana, variables); futuro: calendario conocido (días,24,6).
    # y/mask: etiquetas y presencia original (días,24); js: índices de esos días.
    assert variante in c['variantes']
    z, obs, cal = preparar(df, prep)
    n = 1 if variante == 'base' else len(prep['columnas'])
    # Valor objetivo primero, luego otras mediciones, máscaras y calendario.
    entradas = np.concatenate([z[:, :n], obs[:, :n], cal], axis=1)
    js = dias_objetivo(df, c, bloque)
    y_original = df[objetivo].to_numpy(dtype='float64')
    y = (y_original - prep['media'][objetivo]) / prep['escala'][objetivo]
    return dict(
        x=np.stack([entradas[j-c['ventana']:j] for j in js]),
        futuro=np.stack([cal[j:j+24] for j in js]),
        y=np.stack([y[j:j+24] for j in js]).astype('float32'),
        mask=np.stack([obs[j:j+24, 0] for j in js]).astype(bool),
        js=js,
    )


def auditar(df, estacion, c):
    """Resume cobertura, faltantes, ceros y extremos por variable y bloque.

    Devuelve una tabla descriptiva; no corrige ni elimina episodios difíciles.
    La máxima racha faltante ayuda a interpretar las entradas rellenadas.
    """
    filas = []
    numericas = [x.removesuffix('_observed') for x in df if x.endswith('_observed')]
    for bloque in particiones(c).itertuples():
        sub = df.loc[(df.index >= bloque.inicio) & (df.index < bloque.fin_exclusivo)]
        for col in numericas:
            s = sub[col]
            faltan = s.isna()
            rachas = faltan.groupby((~faltan).cumsum()).sum()
            filas.append(dict(estacion=estacion, bloque=bloque.bloque, variable=col,
                horas=len(s), observados=int(s.notna().sum()), faltantes=int(faltan.sum()),
                ceros=int(s.eq(0).sum()), negativos=int(s.lt(0).sum()),
                minimo=s.min(), p01=s.quantile(.01), mediana=s.median(),
                p99=s.quantile(.99), maximo=s.max(), media=s.mean(),
                mayor_racha_faltante=int(rachas.max())))
    return pd.DataFrame(filas)
