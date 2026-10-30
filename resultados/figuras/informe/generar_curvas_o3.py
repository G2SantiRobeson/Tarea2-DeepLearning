"""Compone las curvas originales de O3 sin entrenar ni recalcular métricas."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def generar_figura():
    """Lee doce CSV y sus JSON, verifica las épocas y guarda la figura a 300 DPI."""
    resultados = Path(__file__).resolve().parents[2]
    estaciones = ['Aotizhongxin', 'Changping', 'Dingling']
    configuraciones = [('base', 2026), ('meteo', 2026), ('meteo', 2027), ('meteo', 2028)]
    titulos = ['Base\nSemilla 2026', 'Meteorológica\nSemilla 2026',
               'Meteorológica\nSemilla 2027', 'Meteorológica\nSemilla 2028']
    esperadas = [[25, 25, 25, 24], [23, 23, 24, 24], [24, 25, 24, 23]]
    salida = Path(__file__).with_name('Figura_O3_Curvas_Aprendizaje.png')
    # Comprobar que las fuentes leídas permanecen intactas tras generar la figura.
    fuentes = []
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 14,
        'axes.labelsize': 14, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.formatter.useoffset': False, 'path.simplify': False,
    })
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.subplots_adjust(left=.06, right=.985, bottom=.10, top=.85,
                        wspace=.36, hspace=.57)
    for fila, estacion in enumerate(estaciones):
        seleccion = resultados / 'modelos' / f'{estacion}_ozone_seleccion.json'
        fuentes.append((seleccion, hashlib.sha256(seleccion.read_bytes()).digest()))
        assert json.loads(seleccion.read_text(encoding='utf-8'))['principal'] == 'meteo'
        for columna, (variante, semilla) in enumerate(configuraciones):
            nombre = f'{estacion}_ozone_lstm_{variante}_{semilla}'
            csv = resultados / 'metricas' / f'{nombre}_curva.csv'
            metadatos = resultados / 'modelos' / f'{nombre}.json'
            for fuente in (csv, metadatos):
                fuentes.append((fuente, hashlib.sha256(fuente.read_bytes()).digest()))
            curva = pd.read_csv(csv, float_precision='round_trip')
            registro = json.loads(metadatos.read_text(encoding='utf-8'))
            assert list(curva.columns) == ['epoca', 'mse_train', 'mse_val']
            assert np.isfinite(curva.to_numpy()).all()
            assert np.array_equal(curva.epoca, np.arange(1, len(curva) + 1))
            assert len(curva) == registro['epocas']
            recuperada = registro['mejor_epoca']
            # El gráfico original marca el mínimo del MSE ya registrado en Validation.
            assert recuperada == curva.loc[curva.mse_val.idxmin(), 'epoca']
            if recuperada != esperadas[fila][columna]:
                print(f'DISCREPANCIA {nombre}: visual={esperadas[fila][columna]}; '
                      f'registro={recuperada}. Se prioriza el registro.')
            ax = axes[fila, columna]
            train, = ax.plot(curva.epoca, curva.mse_train, color='#1f77b4',
                             lw=1.8, label='Train')
            val, = ax.plot(curva.epoca, curva.mse_val, color='#ff7f0e',
                           lw=1.8, label='Validation')
            linea = ax.axvline(recuperada, color='gray', ls='--', lw=1.6,
                               label='Época recuperada')
            ax.set(xlabel='Época', ylabel='MSE normalizado')
            # Conservar las escalas lineales y márgenes del 5 % de las figuras originales.
            ax.margins(x=.05, y=.05)
            ax.text(.50, .94, f'Época recuperada: {recuperada}', transform=ax.transAxes,
                    ha='center', va='top', fontsize=12, color='#555555')
            assert np.array_equal(train.get_xdata(), curva.epoca.to_numpy())
            assert np.array_equal(train.get_ydata(), curva.mse_train.to_numpy())
            assert np.array_equal(val.get_xdata(), curva.epoca.to_numpy())
            assert np.array_equal(val.get_ydata(), curva.mse_val.to_numpy())
            assert list(linea.get_xdata()) == [recuperada, recuperada]
            if fila == 0:
                posicion = ax.get_position()
                fig.text((posicion.x0 + posicion.x1) / 2, .975, titulos[columna],
                         ha='center', va='top', fontsize=16)
            print(f'{nombre}: {len(curva)} épocas; recuperada={recuperada}')
        posicion = axes[fila, 0].get_position()
        fig.text(posicion.x0, posicion.y1 + .018,
                 f'({chr(97 + fila)}) {estacion}', fontsize=16,
                 fontweight='bold', ha='left', va='bottom')
    fig.legend(handles=axes[0, 0].get_lines(), loc='lower center',
               bbox_to_anchor=(.5, .018), ncol=3, frameon=False, fontsize=15)
    assert axes.shape == (3, 4) and all(len(ax.lines) == 3 for ax in axes.flat)
    fig.savefig(salida, dpi=300, facecolor='white')
    plt.close(fig)
    assert all(hashlib.sha256(p.read_bytes()).digest() == h for p, h in fuentes)
    with Image.open(salida) as imagen:
        print(f'{salida}\nDimensiones: {imagen.size}; DPI: {imagen.info.get("dpi")}')
    print(f'Fuentes verificadas sin cambios: {len(fuentes)} archivos CSV/JSON.')
    return salida


if __name__ == '__main__':
    generar_figura()
