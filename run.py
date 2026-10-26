"""Punto de entrada: un caso, una etapa o los seis casos con el mismo flujo."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG, DATOS, RESULTADOS, configuracion
from src.datos import cargar, particiones, ajustar_preparacion, dias_objetivo, ventanas, auditar
from src.modelos_naive import predecir_naive
from src.modelo_sarima import ajustar, pronosticar
from src.modelo_lstm import cargar_modelo
from src.entrenamiento import entrenar, predecir
from src.evaluacion import tabla_predicciones, resumir, verificar_comparabilidad, metricas


def guardar_json(ruta, valor):
    """Guarda configuración o información del ajuste en un archivo legible."""
    ruta.write_text(json.dumps(valor, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def preparar_salida(out, c):
    """Crea las carpetas de salida y comprueba que datos y configuración coincidan.

    out es la carpeta elegida; c contiene los parámetros de config.py. Si ya
    existen resultados con otro protocolo, se exige usar una carpeta diferente.
    """
    for sub in ['modelos', 'metricas', 'predicciones', 'figuras']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    protocolo = dict(config=c, sha256_datos=hashlib.sha256(DATOS.read_bytes()).hexdigest())
    ruta = out / 'configuracion.json'
    if ruta.exists():
        assert json.loads(ruta.read_text(encoding='utf-8')) == protocolo, (
            'Los datos/configuración cambiaron: usa --salida con otra carpeta')
    else:
        guardar_json(ruta, protocolo)
    particiones(c).to_csv(out / 'metricas' / 'particiones.csv', index=False)
    if not (out / 'entorno.json').exists():
        guardar_json(out / 'entorno.json', dict(
            python=platform.python_version(), sistema=platform.platform(),
            procesador=platform.processor(), cpu_logicas=os.cpu_count(),
            hilos_torch=c['hilos'], dispositivo='CPU',
            versiones={p: importlib.metadata.version(p) for p in
                       ['torch', 'numpy', 'pandas', 'scipy', 'statsmodels', 'matplotlib']}))


def ejecutar_caso(estacion, objetivo, etapa, c, out):
    """Ejecuta una etapa para una estación y un contaminante y guarda su evidencia.

    El orden de 'todo' es preparación, Naive, SARIMA y LSTM. Cada bloque de
    código llama directamente al módulo responsable, sin un sistema de plugins.
    """
    caso = f'{estacion}_{objetivo}'
    df = cargar(DATOS, estacion, c)
    prep = ajustar_preparacion(df, objetivo, c)
    guardar_json(out / 'modelos' / f'{caso}_preparacion.json', prep)
    if etapa in ['todo', 'auditoria']:
        auditar(df, estacion, c).to_csv(out / 'metricas' / f'{estacion}_auditoria.csv', index=False)
        # Respaldo compacto: entradas causales y máscaras; los objetivos originales se conservan.
        cols = prep['columnas']
        copia = df[cols].copy()
        for col in cols:
            copia[col + '_observed'] = copia[col].notna()
            copia[col + '_entrada'] = copia[col].ffill().fillna(0)
        copia.to_csv(out / 'modelos' / f'{caso}_preparacion.csv.gz', compression='gzip')
    if etapa == 'auditoria':
        return

    def guardar_pred(nombre, modelo, variante, semilla, bloque, pred, js):
        """Asocia las 24 predicciones a su origen y día objetivo y las guarda."""
        tabla = tabla_predicciones(df, objetivo, estacion, modelo, variante, semilla, bloque, js, pred)
        tabla.to_csv(out / 'predicciones' / f'{caso}_{nombre}_{bloque}.csv.gz', index=False, compression='gzip')

    if etapa in ['todo', 'naive']:
        for rezago in [24, 168]:
            inicio = time.perf_counter()
            for bloque in ['val', 'test']:
                js = dias_objetivo(df, c, bloque)
                guardar_pred(f'naive_{rezago}', 'Naive', str(rezago), 'sin_ajuste', bloque,
                             predecir_naive(df, objetivo, js, rezago), js)
            guardar_json(out / 'modelos' / f'{caso}_naive_{rezago}.json',
                         dict(segundos_prediccion_y_guardado=time.perf_counter()-inicio, parametros=0))
    if etapa in ['todo', 'sarima']:
        ruta = out / 'modelos' / f'{caso}_sarima.pkl'
        diag = ruta.with_suffix('.json')
        if not (ruta.exists() and diag.exists()):
            print(f'{caso}: ajustando SARIMA', flush=True)
            diagnostico, residuos = ajustar(df, objetivo, c, prep, ruta)
            guardar_json(diag, diagnostico)
            residuos.to_csv(out / 'metricas' / f'{caso}_residuos.csv.gz', index=False, compression='gzip')
        for bloque in ['val', 'test']:
            js = dias_objetivo(df, c, bloque)
            destino = out / 'predicciones' / f'{caso}_sarima_{bloque}.csv.gz'
            if not destino.exists():
                inicio = time.perf_counter()
                pred = pronosticar(df, objetivo, js, prep, c, ruta)
                guardar_pred('sarima', 'SARIMA', 'estacional24', 'unico', bloque, pred, js)
                guardar_json(out / 'modelos' / f'{caso}_sarima_tiempo_{bloque}.json',
                             dict(segundos_prediccion=time.perf_counter()-inicio))
    if etapa in ['todo', 'lstm']:
        puntuaciones = {}

        def entrenar_variante(variante, semilla):
            """Entrena o carga una red y devuelve su MAE de Validation original.

            Este valor permite elegir entradas antes de acceder a Test.
            """
            nombre = f'lstm_{variante}_{semilla}'
            ruta = out / 'modelos' / f'{caso}_{nombre}.pt'
            train = ventanas(df, objetivo, variante, c, prep, 'train')
            val = ventanas(df, objetivo, variante, c, prep, 'val')
            if not (ruta.exists() and ruta.with_suffix('.json').exists()):
                print(f'{caso}: entrenando {variante}, semilla {semilla}', flush=True)
                modelo, curva, info = entrenar(train, val, c, semilla, ruta, prep)
                curva.to_csv(out / 'metricas' / f'{caso}_{nombre}_curva.csv', index=False)
                guardar_json(ruta.with_suffix('.json'), info)
            else:
                modelo, _ = cargar_modelo(ruta)
            inicio = time.perf_counter()
            pred = predecir(modelo, val, c['batch_size']) * prep['escala'][objetivo] + prep['media'][objetivo]
            guardar_pred(nombre, 'LSTM', variante, semilla, 'val', pred, val['js'])
            guardar_json(out / 'modelos' / f'{caso}_{nombre}_tiempo_val.json',
                         dict(segundos_prediccion=time.perf_counter()-inicio))
            real = np.stack([df[objetivo].to_numpy()[j:j+24] for j in val['js']])
            return metricas(real, pred, val['mask'])['MAE']

        for variante in c['variantes']:
            puntuaciones[variante] = entrenar_variante(variante, c['semillas'][0])
        # Empate exacto: primera variante (base). Solo semilla compartida para elegir entradas.
        principal = min(puntuaciones, key=puntuaciones.get)
        seleccion = dict(principal=principal, criterio=c['seleccion_variante'],
                         mae_val_semilla_compartida=puntuaciones, semillas=c['semillas'])
        guardar_json(out / 'modelos' / f'{caso}_seleccion.json', seleccion)
        for semilla in c['semillas'][1:]:
            entrenar_variante(principal, semilla)
        # TEST se construye y predice una vez fijadas entradas, épocas y semillas.
        ejecuciones = [(v, c['semillas'][0]) for v in c['variantes']]
        ejecuciones += [(principal, s) for s in c['semillas'][1:]]
        for variante, semilla in ejecuciones:
            nombre = f'lstm_{variante}_{semilla}'
            modelo, ckpt = cargar_modelo(out / 'modelos' / f'{caso}_{nombre}.pt')
            assert ckpt['config'] == c and ckpt['preparacion'] == prep
            test = ventanas(df, objetivo, variante, c, prep, 'test')
            inicio = time.perf_counter()
            pred = predecir(modelo, test, c['batch_size']) * prep['escala'][objetivo] + prep['media'][objetivo]
            guardar_pred(nombre, 'LSTM', variante, semilla, 'test', pred, test['js'])
            guardar_json(out / 'modelos' / f'{caso}_{nombre}_tiempo_test.json',
                         dict(segundos_prediccion=time.perf_counter()-inicio))


def evaluar(out):
    """Recalcula tablas desde predicciones guardadas en out, sin entrenar modelos.

    Devuelve la tabla horaria para graficar. Primero exige que todos los modelos
    de cada caso y bloque evalúen exactamente las mismas observaciones.
    """
    archivos = sorted((out / 'predicciones').glob('*.csv.gz'))
    if not archivos:
        raise FileNotFoundError('Primero ejecuta naive, sarima, lstm o todo')
    tabla = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)
    verificar_comparabilidad(tabla)
    resumen, horizontes = resumir(tabla)
    principales = {}
    for archivo in (out / 'modelos').glob('*_seleccion.json'):
        principales[archivo.stem.removesuffix('_seleccion')] = json.loads(archivo.read_text(encoding='utf-8'))['principal']
    resumen['principal'] = [r.modelo == 'LSTM' and r.variante == principales.get(f'{r.estacion}_{r.objetivo}')
                            for r in resumen.itertuples()]
    resumen.to_csv(out / 'metricas' / 'comparacion.csv', index=False)
    horizontes.to_csv(out / 'metricas' / 'por_horizonte.csv', index=False)
    seeds = resumen.loc[resumen.principal]
    if len(seeds):
        variabilidad = seeds.groupby(['estacion', 'objetivo', 'variante', 'bloque']).agg(
            semillas=('ejecucion', 'nunique'), MAE_media=('MAE', 'mean'), MAE_std=('MAE', 'std'),
            MAE_min=('MAE', 'min'), MAE_max=('MAE', 'max'),
            RMSE_media=('RMSE', 'mean'), RMSE_std=('RMSE', 'std'))
        variabilidad.to_csv(out / 'metricas' / 'variabilidad.csv')
    # Un inventario auditable distingue una ejecución parcial del alcance completo.
    guardar_json(out / 'metricas' / 'inventario.json', dict(
        casos=int(resumen[['estacion', 'objetivo']].drop_duplicates().shape[0]),
        referencias_naive=int(resumen.loc[resumen.modelo.eq('Naive'), ['estacion', 'objetivo', 'variante']].drop_duplicates().shape[0]),
        ajustes_sarima=len(list((out / 'modelos').glob('*_sarima.pkl'))),
        entrenamientos_lstm=len(list((out / 'modelos').glob('*.pt'))),
        filas_prediccion=len(tabla), comparabilidad_verificada=True))
    return tabla


def main():
    """Lee la etapa y las opciones de caso/carpeta y ejecuta el flujo elegido."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('etapa', choices=['auditoria', 'naive', 'sarima', 'lstm', 'evaluar', 'figuras', 'todo'],
                        help='auditoria prepara datos; evaluar usa predicciones guardadas; todo ejecuta el flujo completo')
    parser.add_argument('--estacion', choices=CONFIG['estaciones'], help='Omitir para las tres estaciones')
    parser.add_argument('--objetivo', choices=CONFIG['objetivos'], help='Omitir para ambos contaminantes')
    parser.add_argument('--salida', type=Path, help='Carpeta de resultados; usar otra para nuevos experimentos')
    args = parser.parse_args()
    c = configuracion()
    out = args.salida or RESULTADOS
    preparar_salida(out, c)
    estaciones = [args.estacion] if args.estacion else c['estaciones']
    objetivos = [args.objetivo] if args.objetivo else c['objetivos']
    if args.etapa not in ['evaluar', 'figuras']:
        for estacion in estaciones:
            for objetivo in objetivos:
                ejecutar_caso(estacion, objetivo, args.etapa, c, out)
    if args.etapa in ['todo', 'evaluar', 'figuras']:
        tabla = evaluar(out)
        if args.etapa != 'evaluar':
            from src.graficos import graficar
            graficar(out, tabla)
    print(f'Etapa {args.etapa} terminada. Resultados: {out}', flush=True)


if __name__ == '__main__':
    main()
