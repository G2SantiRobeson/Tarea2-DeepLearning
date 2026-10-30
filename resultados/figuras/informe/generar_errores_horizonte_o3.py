"""Compone errores de O3 por horizonte desde el CSV original, sin recalcularlos."""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def generar_figura():
    """Lee los errores guardados y devuelve el PNG de tres estaciones a 300 DPI."""
    resultados = Path(__file__).resolve().parents[2]
    archivo = resultados / 'metricas' / 'por_horizonte.csv'
    huella = hashlib.sha256(archivo.read_bytes()).digest()
    datos = pd.read_csv(archivo, dtype={'ejecucion': str, 'variante': str}, float_precision='round_trip')
    estaciones = ['Aotizhongxin', 'Changping', 'Dingling']
    columnas = [('val', 'MAE'), ('val', 'RMSE'), ('test', 'MAE'), ('test', 'RMSE')]
    # Se mantienen los colores de las figuras originales; los estilos distinguen cruces.
    alternativas = [
        ('Naive', '24', 'sin_ajuste', 'Naive diario', '#d62728', '--'),
        ('Naive', '168', 'sin_ajuste', 'Naive semanal', '#2ca02c', ':'),
        ('SARIMA', 'estacional24', 'unico', 'SARIMA estacional', '#9467bd', '-.'),
        ('LSTM', 'base', '2026', 'LSTM base — 2026', '#1f77b4', '-'),
        ('LSTM', 'meteo', '2026', 'LSTM meteorológica — 2026', '#ff7f0e', '-'),
    ]
    seleccion = datos.loc[
        datos.objetivo.eq('ozone') & datos.estacion.isin(estaciones)
        & (~datos.modelo.eq('LSTM') | datos.ejecucion.eq('2026'))
    ]
    assert seleccion.objetivo.eq('ozone').all()
    assert seleccion.loc[seleccion.modelo.eq('LSTM'), 'ejecucion'].eq('2026').all()
    series = {}
    for estacion in estaciones:
        for bloque, metrica in columnas:
            for modelo, variante, ejecucion, *_ in alternativas:
                curva = seleccion.loc[
                    seleccion.estacion.eq(estacion) & seleccion.bloque.eq(bloque)
                    & seleccion.modelo.eq(modelo) & seleccion.variante.eq(variante)
                    & seleccion.ejecucion.eq(ejecucion)
                ].sort_values('horizonte')
                assert np.array_equal(curva.horizonte.to_numpy(), np.arange(1, 25))
                assert np.isfinite(curva[metrica]).all() and curva[metrica].ge(0).all()
                series[estacion, bloque, metrica, modelo, variante] = curva
    # Contrastar los extremos de MAE Test con las referencias redondeadas.
    referencias = {
        'Aotizhongxin': [(7.755, 28.105), (13.341, 23.271), (13.435, 23.394)],
        'Changping': [(9.210, 28.661), (14.088, 20.275), (12.188, 20.741)],
        'Dingling': [(7.568, 29.164), (13.538, 24.080), (13.007, 23.468)],
    }
    for estacion in estaciones:
        for (modelo, variante), referencia in zip(
            [('SARIMA', 'estacional24'), ('LSTM', 'base'), ('LSTM', 'meteo')],
            referencias[estacion],
        ):
            curva = series[estacion, 'test', 'MAE', modelo, variante]
            extremos = curva.MAE.iloc[[0, -1]].to_numpy()
            assert np.all(np.abs(extremos - referencia) <= .0005)
            print(f'{estacion} {modelo} {variante}: MAE Test h1/h24 = {extremos}')
    # Conservar h5 incluso cuando hay menos observaciones válidas.
    for bloque, observados in [('val', 106), ('test', 154)]:
        h5 = seleccion.loc[seleccion.estacion.eq('Aotizhongxin')
                           & seleccion.bloque.eq(bloque) & seleccion.horizonte.eq(5)]
        assert len(h5) == 5
        assert h5.n_observados.eq(observados).all() and h5.n_total.eq(219).all()
        print(f'Aotizhongxin {bloque} h5: {observados}/219 observaciones; cinco modelos conservados.')
    # Un mismo rango por métrica en las tres estaciones y ambos conjuntos.
    limites = {
        metrica: max(curva[metrica].max() for clave, curva in series.items() if clave[2] == metrica) * 1.06
        for metrica in ['MAE', 'RMSE']
    }
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 14,
        'axes.labelsize': 14, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.subplots_adjust(left=.055, right=.985, bottom=.15, top=.90, wspace=.31, hspace=.57)
    for fila, estacion in enumerate(estaciones):
        for columna, (bloque, metrica) in enumerate(columnas):
            ax = axes[fila, columna]
            for modelo, variante, ejecucion, etiqueta, color, estilo in alternativas:
                curva = series[estacion, bloque, metrica, modelo, variante]
                linea, = ax.plot(curva.horizonte, curva[metrica], color=color,
                                 ls=estilo, lw=1.8, label=etiqueta)
                # Verificación directa de los 24 valores trazados, sin suavizado.
                assert np.array_equal(linea.get_ydata(), curva[metrica].to_numpy())
                assert np.array_equal(linea.get_xdata(), curva.horizonte.to_numpy())
            ax.set(xlabel='Horizonte (horas)', ylabel=f'{metrica} (µg/m³)',
                   xlim=(.5, 24.5), ylim=(0, limites[metrica]),
                   xticks=[1, 4, 8, 12, 16, 20, 24])
            ax.grid(axis='y', color='#e6e6e6', lw=.6)
            ax.set_axisbelow(True)
            if fila == 0:
                conjunto = 'Validation' if bloque == 'val' else 'Test'
                ax.set_title(f'{conjunto} — {metrica}', fontsize=16, pad=42)
            assert len(ax.lines) == 5
        posicion = axes[fila, 0].get_position()
        fig.text(posicion.x0, posicion.y1 + .018,
                 f'({chr(97 + fila)}) {estacion} · O₃',
                 fontsize=16, fontweight='bold', ha='left', va='bottom')
    fig.legend(handles=axes[0, 0].get_lines(), loc='lower center', ncol=3,
               bbox_to_anchor=(.5, .025), frameon=False, fontsize=15,
               handlelength=3, columnspacing=2.4, labelspacing=.8)
    salida = Path(__file__).with_name('Figura_O3_Errores_Horizonte.png')
    fig.savefig(salida, dpi=300, facecolor='white')
    plt.close(fig)
    assert hashlib.sha256(archivo.read_bytes()).digest() == huella
    with Image.open(salida) as imagen:
        print(f'{salida}\nDimensiones: {imagen.size}; DPI: {imagen.info.get("dpi")}')
    print('Verificado: 3 estaciones x 4 paneles x 5 modelos x 24 horizontes.')
    print('Solo LSTM 2026. CSV original sin cambios. Rangos comunes por metrica:', limites)
    return salida


if __name__ == '__main__':
    generar_figura()
