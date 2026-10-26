"""Única configuración del estudio. Cambiarla exige otra carpeta de resultados."""
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATOS = BASE / 'data' / 'beijing_consolidated.csv'
RESULTADOS = BASE / 'resultados'
CONFIG = {
    'inicio': '2013-03-01', 'inicio_val': '2015-12-19',
    'inicio_test': '2016-07-25', 'fin': '2017-03-01',
    'estaciones': ['Aotizhongxin', 'Changping', 'Dingling'],
    'objetivos': ['pm25', 'ozone'],
    'ventana': 48, 'horizonte': 24,
    'meteorologia': ['temp_C', 'pressure_hPa', 'dewpoint_C', 'rain_mm', 'wind_ms'],
    'variantes': ['base', 'meteo'], 'semillas': [2026, 2027, 2028],
    'ocultas': 16, 'capas': 1, 'lr': 0.003, 'batch_size': 256,
    'epocas': 25, 'paciencia': 4, 'clip': 1.0, 'hilos': 2,
    'sarima_order': [1, 0, 0], 'sarima_seasonal': [1, 0, 0, 24],
    'sarima_maxiter': 100,
    'teacher_forcing': False,
    'seleccion_epoca': 'MSE_validation_normalizado',
    'seleccion_variante': 'MAE_validation_original',
}


def configuracion():
    """Devuelve una copia de los parámetros del estudio para run.py y el notebook."""
    return dict(CONFIG)
