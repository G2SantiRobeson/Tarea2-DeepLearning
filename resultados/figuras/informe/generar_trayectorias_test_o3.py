"""Compone las primeras 168 horas originales de O3 en Test sin alterar series."""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def generar_figura():
    """Lee CSV originales y devuelve un PNG con tres paneles y huecos conservados."""
    resultados = Path(__file__).resolve().parents[2]
    estaciones = ['Aotizhongxin', 'Changping', 'Dingling']
    alternativas = [
        ('Naive', '24', 'sin_ajuste', 'Naive diario', '#d62728', '--'),
        ('Naive', '168', 'sin_ajuste', 'Naive semanal', '#2ca02c', ':'),
        ('SARIMA', 'estacional24', 'unico', 'SARIMA estacional', '#9467bd', '-.'),
        ('LSTM', 'base', '2026', 'LSTM base — 2026', '#1f77b4', '-'),
        ('LSTM', 'meteo', '2026', 'LSTM meteorológica — 2026', '#ff7f0e', '-'),
    ]
    originales = sorted((resultados / 'predicciones').glob('*ozone*test.csv.gz'))
    originales += [resultados / 'figuras' / f'{e}_ozone_test_trayectorias.png' for e in estaciones]
    huellas = {p: hashlib.sha256(p.read_bytes()).digest() for p in originales}
    tabla = pd.concat([
        pd.read_csv(p, dtype={'variante': str, 'ejecucion': str},
                    parse_dates=['hora_objetivo'], float_precision='round_trip')
        for p in originales if p.name.endswith('.csv.gz')
    ], ignore_index=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 14,
        'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 13,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'path.simplify': False,
    })
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.subplots_adjust(left=.085, right=.985, bottom=.13, top=.95, hspace=.43)
    resumen = []
    for fila, estacion in enumerate(estaciones):
        caso = tabla.loc[tabla.estacion.eq(estacion) & tabla.objetivo.eq('ozone') & tabla.bloque.eq('test')]
        assert not caso.empty and caso.objetivo.eq('ozone').all()
        inicio = caso.hora_objetivo.min()
        # Misma selección temporal que src/graficos.py: mínimo de Test + siete días.
        sub = caso.loc[caso.hora_objetivo < inicio + pd.Timedelta(hours=168)]
        horas = pd.date_range(inicio, periods=168, freq='h')
        real = sub.drop_duplicates('hora_objetivo').sort_values('hora_objetivo')
        assert np.array_equal(real.hora_objetivo.to_numpy(), horas.to_numpy())
        assert inicio == pd.Timestamp('2016-07-25 00:00:00')
        assert horas[-1] == pd.Timestamp('2016-07-31 23:00:00')
        ax = axes[fila]
        observada, = ax.plot(real.hora_objetivo, real.observacion, color='black',
                            lw=2.2, zorder=5, label='Observado')
        assert np.array_equal(observada.get_ydata(), real.observacion.to_numpy(), equal_nan=True)
        # Comprobar que incluso las otras semillas registran las mismas observaciones.
        for _, grupo in sub.groupby(['modelo', 'variante', 'ejecucion']):
            grupo = grupo.sort_values('hora_objetivo')
            assert np.array_equal(grupo.hora_objetivo.to_numpy(), horas.to_numpy())
            assert np.array_equal(grupo.observacion.to_numpy(), real.observacion.to_numpy(), equal_nan=True)
        for modelo, variante, ejecucion, etiqueta, color, estilo in alternativas:
            curva = sub.loc[sub.modelo.eq(modelo) & sub.variante.eq(variante)
                            & sub.ejecucion.eq(ejecucion)].sort_values('hora_objetivo')
            assert len(curva) == 168
            assert curva.hora_objetivo.is_unique
            assert np.array_equal(curva.hora_objetivo.to_numpy(), horas.to_numpy())
            if modelo == 'LSTM':
                assert curva.ejecucion.eq('2026').all()
            linea, = ax.plot(curva.hora_objetivo, curva.estimacion, label=etiqueta,
                             color=color, ls=estilo, lw=1.45)
            assert np.array_equal(linea.get_xdata(), curva.hora_objetivo.to_numpy())
            assert np.array_equal(linea.get_ydata(), curva.estimacion.to_numpy(), equal_nan=True)
        assert len(ax.lines) == 6
        ax.set_title(f'({chr(97 + fila)}) {estacion}', loc='left', fontsize=16, fontweight='bold', pad=12)
        ax.set(ylabel='O3 (µg/m³)', xlabel='Fecha y hora')
        ax.set_xticks(pd.date_range(inicio.normalize(), horas[-1].normalize(), freq='D'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y\n%H:%M'))
        ax.set_xlim(inicio - pd.Timedelta(hours=1), horas[-1] + pd.Timedelta(hours=1))
        ax.margins(y=.07)
        ax.grid(axis='y', color='#e6e6e6', lw=.6)
        ax.set_axisbelow(True)
        resumen.append(f'{estacion}: {inicio} a {horas[-1]}; 168 horas; '
                       f'{real.observacion.isna().sum()} observaciones ausentes conservadas.')
    fig.legend(handles=axes[0].get_lines(), loc='lower center', bbox_to_anchor=(.5, .025),
               ncol=3, frameon=False, fontsize=14, handlelength=3, labelspacing=.9, columnspacing=2.2)
    salida = Path(__file__).with_name('Figura_O3_Trayectorias_Test.png')
    fig.savefig(salida, dpi=300, facecolor='white')
    plt.close(fig)
    assert all(hashlib.sha256(p.read_bytes()).digest() == h for p, h in huellas.items())
    with Image.open(salida) as imagen:
        print(f'{salida}\nDimensiones: {imagen.size}; DPI: {imagen.info.get("dpi")}')
    print('\n'.join(resumen))
    print(f'Verificados 3 paneles, 6 series por panel y {len(huellas)} originales sin cambios.')
    return salida


if __name__ == '__main__':
    generar_figura()
