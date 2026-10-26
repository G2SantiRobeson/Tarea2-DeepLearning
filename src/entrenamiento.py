"""Ciclo PyTorch del Lab 8, con pérdida enmascarada para Beijing."""
import copy
import random
import time
import numpy as np
import pandas as pd
import torch
from .modelo_lstm import Seq2Seq, cargar_modelo


def mse_enmascarado(predicciones, objetivos, mascara):
    """Devuelve la pérdida escalar MSE de etiquetas observadas del minilote.

    Los tres tensores tienen forma (B,24). Se promedian los errores cuadrados
    normalizados; la máscara excluye faltantes y mantiene los ceros observados.
    """
    # Indexar ANTES de restar evita propagar NaN incluso donde la máscara es falsa.
    if not bool(mascara.any()):
        return predicciones.sum() * 0
    return ((predicciones[mascara] - objetivos[mascara]) ** 2).mean()


def tensores(datos):
    """Convierte los cuatro arreglos de una partición a tensores PyTorch.

    Conserva los tipos: valores float32 y máscara booleana. Los índices js solo
    identifican fechas y no se entregan a la red ni participan en el gradiente.
    """
    return {k: torch.as_tensor(datos[k]) for k in ['x', 'futuro', 'y', 'mask']}


@torch.no_grad()
def predecir(modelo, datos, batch_size=256):
    """Devuelve (días,24) en escala normalizada, sin actualizar pesos.

    Procesar minilotes limita la memoria. no_grad() evita construir el grafo de
    gradientes, necesario en entrenamiento pero innecesario al evaluar.
    """
    modelo.eval()
    datos_tensores = tensores(datos)
    predicciones = []
    for inicio in range(0, len(datos_tensores['x']), batch_size):
        fin = inicio + batch_size
        historia = datos_tensores['x'][inicio:fin]
        calendario_futuro = datos_tensores['futuro'][inicio:fin]
        predicciones.append(modelo(historia, calendario_futuro))
    return torch.cat(predicciones).numpy()


def entrenar(train, val, c, semilla, ruta, prep):
    """Aprende con Train y recupera la época de menor MSE de Validation.

    Recibe las ventanas normalizadas y configuración, guarda pesos y preparación
    en ruta y devuelve el modelo recuperado, curva de aprendizaje y costo medido.
    Validation mide la pérdida al final de cada época y no actualiza parámetros.
    """
    inicio = time.perf_counter()
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.set_num_threads(c['hilos'])
    torch.use_deterministic_algorithms(True)
    arquitectura = dict(entradas=train['x'].shape[2], ocultas=c['ocultas'])
    modelo = Seq2Seq(**arquitectura)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=c['lr'])
    datos_tensores = tensores(train)
    mejor, espera, historia, pesos = float('inf'), 0, [], None
    assert train['mask'].any() and val['mask'].any()
    for epoca in range(1, c['epocas'] + 1):
        modelo.train()
        # Una época recorre las ventanas de Train. Se barajan ejemplos completos;
        # dentro de cada ventana se conserva siempre el orden de sus 48 horas.
        orden = torch.randperm(len(datos_tensores['x']))
        for indices in orden.split(c['batch_size']):
            mask = datos_tensores['mask'][indices]
            if not bool(mask.any()):
                continue  # Un lote sin etiquetas no aporta una actualización.
            # Forward: Encoder y Decoder generan las 24 salidas del minilote.
            pred = modelo(datos_tensores['x'][indices], datos_tensores['futuro'][indices])
            loss = mse_enmascarado(pred, datos_tensores['y'][indices], mask)
            # Borrar gradientes del lote anterior antes de acumular los actuales.
            optimizer.zero_grad()
            loss.backward()  # Autograd aplica BPTT por Encoder y Decoder completos.
            # Limitar gradientes explosivos antes de que Adam actualice los pesos.
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), c['clip'])
            optimizer.step()
        # Pérdidas con los mismos pesos finales de la época, no promedio de minilotes.
        p_train = predecir(modelo, train, c['batch_size'])
        p_val = predecir(modelo, val, c['batch_size'])
        ltrain = float(np.mean((p_train[train['mask']] - train['y'][train['mask']]) ** 2))
        lval = float(np.mean((p_val[val['mask']] - val['y'][val['mask']]) ** 2))
        if not np.isfinite(lval + ltrain):
            raise ValueError('Pérdida no finita; revisar el experimento')
        historia.append(dict(epoca=epoca, mse_train=ltrain, mse_val=lval))
        if lval < mejor:
            # Copiar los mejores pesos: entrenar la siguiente época no debe
            # sobrescribir el respaldo que luego recuperará Early Stopping.
            mejor, espera, mejor_epoca = lval, 0, epoca
            pesos = copy.deepcopy(modelo.state_dict())
        else:
            espera += 1
        if espera >= c['paciencia']:
            break
    duracion = time.perf_counter() - inicio
    torch.save(dict(pesos=pesos, arquitectura=arquitectura, preparacion=prep,
                    config=c, semilla=semilla, mejor_epoca=mejor_epoca,
                    mse_val=mejor), ruta)
    modelo, _ = cargar_modelo(ruta)  # Verifica la reconstrucción antes de evaluar.
    return modelo, pd.DataFrame(historia), dict(
        segundos_ajuste=duracion, epocas=len(historia), mejor_epoca=mejor_epoca,
        parametros=sum(p.numel() for p in modelo.parameters()))
