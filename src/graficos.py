"""Figuras de evidencia; episodios elegidos por fecha, nunca por menor error."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def graficar(out, tabla):
    """Guarda figuras desde predicciones, métricas y curvas existentes en out.

    tabla contiene filas horarias de Validation/Test. Se ilustran las primeras
    168 horas de cada bloque por una regla fija, no por escoger episodios favorables.
    No ajusta modelos ni cambia decisiones a partir de lo observado en Test.
    """
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    for (estacion, objetivo), caso in tabla.groupby(['estacion', 'objetivo']):
        nombre = f'{estacion}_{objetivo}'
        seleccion_path = out / 'modelos' / f'{nombre}_seleccion.json'
        principal = json.loads(seleccion_path.read_text(encoding='utf-8'))['principal'] if seleccion_path.exists() else None
        # Una figura de trayectorias por bloque; se muestran ambas variantes en semilla común.
        for bloque, sub in caso.groupby('bloque'):
            sub = sub.copy()
            sub['hora_objetivo'] = pd.to_datetime(sub.hora_objetivo)
            sub = sub.loc[sub.hora_objetivo < sub.hora_objetivo.min() + pd.Timedelta(days=7)]
            fig, ax = plt.subplots(figsize=(11, 4))
            real = sub.drop_duplicates('hora_objetivo').sort_values('hora_objetivo')
            ax.plot(real.hora_objetivo, real.observacion, color='black', label='Observado', lw=1.6)
            for (modelo, variante, semilla), g in sub.groupby(['modelo', 'variante', 'ejecucion']):
                if modelo == 'LSTM' and str(semilla) != '2026':
                    continue
                ax.plot(g.hora_objetivo, g.estimacion, label=f'{modelo} {variante}', alpha=.8, lw=1)
            ax.set(title=f'{estacion} · {objetivo} · primeras 168 horas de {bloque}', ylabel='µg/m³')
            ax.legend(ncol=3, fontsize=8)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(out / 'figuras' / f'{nombre}_{bloque}_trayectorias.png', dpi=150)
            plt.close(fig)
        errores = pd.read_csv(out / 'metricas' / 'por_horizonte.csv')
        errores = errores.loc[errores.estacion.eq(estacion) & errores.objetivo.eq(objetivo)]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for i, bloque in enumerate(['val', 'test']):
            for (modelo, variante, semilla), g in errores.loc[errores.bloque.eq(bloque)].groupby(['modelo', 'variante', 'ejecucion']):
                if modelo == 'LSTM' and str(semilla) != '2026':
                    continue
                for k, metrica in enumerate(['MAE', 'RMSE']):
                    axes[i, k].plot(g.horizonte, g[metrica], label=f'{modelo} {variante}')
                    axes[i, k].set(title=f'{bloque} · {metrica}', ylabel='µg/m³', xlabel='Horizonte (horas)')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'{estacion} · {objetivo} · error por horizonte')
        fig.tight_layout()
        fig.savefig(out / 'figuras' / f'{nombre}_horizontes.png', dpi=150)
        plt.close(fig)
        curvas = sorted((out / 'metricas').glob(f'{nombre}_lstm_*_curva.csv'))
        if curvas:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
            for ax, archivo in zip(axes.ravel(), curvas):
                curva = pd.read_csv(archivo)
                ax.plot(curva.epoca, curva.mse_train, label='Train')
                ax.plot(curva.epoca, curva.mse_val, label='Validation')
                ax.axvline(curva.loc[curva.mse_val.idxmin(), 'epoca'], color='gray', ls='--', label='Época recuperada')
                ax.set(title=archivo.stem.replace(nombre + '_lstm_', '').replace('_curva', ''),
                       xlabel='Época', ylabel='MSE normalizado')
                ax.legend(fontsize=8)
            fig.suptitle(f'{estacion} · {objetivo} · principal: {principal}')
            fig.tight_layout()
            fig.savefig(out / 'figuras' / f'{nombre}_aprendizaje.png', dpi=150)
            plt.close(fig)
