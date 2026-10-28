"""Compone las doce curvas guardadas de PM2.5, sin ejecutar modelos."""
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
    """Lee CSV/JSON originales y devuelve la ruta del PNG compuesto a 300 DPI."""
    resultados = Path(__file__).resolve().parents[2]
    estaciones = ['Aotizhongxin', 'Changping', 'Dingling']
    configuraciones = [('base', 2026), ('base', 2027), ('base', 2028), ('meteo', 2026)]
    titulos = ['Base — 2026', 'Base — 2027', 'Base — 2028', 'Meteorológica — 2026']
    salida = Path(__file__).with_name('Figura_PM25_Curvas_Aprendizaje.png')
    # Registrar los originales permite comprobar que la composición no los modifica.
    originales = [p for p in resultados.rglob('*') if p.is_file() and salida.parent not in p.parents]
    huellas = {p: hashlib.sha256(p.read_bytes()).digest() for p in originales}
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 14,
        'axes.labelsize': 14, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.formatter.useoffset': False,
    })
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.subplots_adjust(left=.06, right=.985, bottom=.10, top=.90, wspace=.36, hspace=.57)
    resumen = []
    for fila, estacion in enumerate(estaciones):
        principal = json.loads((resultados / 'modelos' / f'{estacion}_pm25_seleccion.json').read_text(encoding='utf-8'))['principal']
        for columna, (variante, semilla) in enumerate(configuraciones):
            nombre = f'{estacion}_pm25_lstm_{variante}_{semilla}'
            curva = pd.read_csv(resultados / 'metricas' / f'{nombre}_curva.csv', float_precision='round_trip')
            registro = json.loads((resultados / 'modelos' / f'{nombre}.json').read_text(encoding='utf-8'))
            assert list(curva.columns) == ['epoca', 'mse_train', 'mse_val']
            assert np.isfinite(curva.to_numpy()).all()
            assert np.array_equal(curva.epoca, np.arange(1, len(curva) + 1))
            assert len(curva) == registro['epocas']
            recuperada = registro['mejor_epoca']
            # La figura original usa el mínimo de Validation; debe coincidir con el JSON.
            assert recuperada == curva.loc[curva.mse_val.idxmin(), 'epoca']
            ax = axes[fila, columna]
            train, = ax.plot(curva.epoca, curva.mse_train, color='#1f77b4', lw=1.8, label='Train')
            val, = ax.plot(curva.epoca, curva.mse_val, color='#ff7f0e', lw=1.8, label='Validation')
            linea = ax.axvline(recuperada, color='gray', ls='--', lw=1.6, label='Época recuperada')
            ax.set(xlabel='Época', ylabel='MSE normalizado', xscale='linear', yscale='linear')
            # Mismos límites automáticos y margen del 5 % que en los paneles originales.
            ax.margins(x=.05, y=.05)
            assert np.array_equal(train.get_ydata(), curva.mse_train.to_numpy())
            assert np.array_equal(val.get_ydata(), curva.mse_val.to_numpy())
            assert list(linea.get_xdata()) == [recuperada, recuperada]
            if fila == 0:
                ax.set_title(titulos[columna], fontsize=16, pad=42)
            resumen.append(f'{nombre}: {len(curva)} épocas; recuperada = {recuperada}')
        posicion = axes[fila, 0].get_position()
        fig.text(posicion.x0, posicion.y1 + .018,
                 f'({chr(97 + fila)}) {estacion} · PM2.5 · principal: {principal}',
                 fontsize=16, fontweight='bold', ha='left', va='bottom')
    fig.legend(handles=axes[0, 0].get_lines(), loc='lower center',
               bbox_to_anchor=(.5, .018), ncol=3, frameon=False, fontsize=15)
    assert axes.shape == (3, 4)
    fig.savefig(salida, dpi=300, facecolor='white')
    plt.close(fig)
    assert all(hashlib.sha256(p.read_bytes()).digest() == huella for p, huella in huellas.items())
    with Image.open(salida) as imagen:
        print(f'{salida}\nDimensiones: {imagen.size}; DPI: {imagen.info.get("dpi")}')
    print('\n'.join(resumen))
    print(f'Originales sin cambios: {len(huellas)} archivos.')
    return salida


if __name__ == '__main__':
    generar_figura()
