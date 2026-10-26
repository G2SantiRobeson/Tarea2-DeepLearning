# Tarea 2 · Pronóstico de calidad del aire en Beijing

Se comparan **Naive diario y semanal, SARIMA y LSTM Seq2Seq** para PM2.5 y O3 en Aotizhongxin, Changping y Dingling: seis casos. A las **23:00 del reloj nominal del CSV** se pronostican las 24 horas del día siguiente.

## Dónde se hace cada cosa

| Archivo | Responsabilidad |
|---|---|
| `src/datos.py` | Cargar los CSV, preparar entradas y construir ventanas |
| `src/modelos_naive.py` | Copiar el perfil del día y de la semana anterior |
| `src/modelo_sarima.py` | Ajustar SARIMA y generar sus pronósticos |
| `src/modelo_lstm.py` | Definir el Encoder, el Decoder y la salida lineal |
| `src/entrenamiento.py` | Entrenar con PyTorch y recuperar la mejor época |
| `src/evaluacion.py` | Calcular MAE, RMSE y cobertura comparable |
| `src/graficos.py` | Generar trayectorias, errores por horizonte y curvas |
| `config.py` | Fechas, ventana, semillas e hiperparámetros |
| `run.py` | Ejecutar una etapa para un caso o para los seis |
| `exploracion.ipynb` | Explorar Train y consultar los resultados guardados |

`data/` contiene los dos CSV necesarios. Los archivos originales del enunciado y las clases están en la carpeta superior y se conservan intactos. Los módulos incluyen comentarios sobre sus entradas, salidas y operaciones importantes.

## Instalar

Se usó Python 3.12, en CPU. Desde esta carpeta, en PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si `.venv` ya existe, reutilizarlo. En Linux/macOS, usar `python3` y `.venv/bin/python`. No se requieren Node, ReportLab ni herramientas de generación de documentos para ejecutar el proyecto.

## Ejecutar

Los resultados definitivos **ya están calculados**. Para preparar y reproducir un caso nuevo, primero ejecutar las dos primeras líneas; para completar los seis casos, usar la tercera:

```powershell
.\.venv\Scripts\python.exe run.py auditoria --salida resultados/reproduccion
.\.venv\Scripts\python.exe run.py todo --estacion Aotizhongxin --objetivo pm25 --salida resultados/reproduccion
.\.venv\Scripts\python.exe run.py todo --salida resultados/reproduccion
```

`auditoria` describe los datos y guarda su preparación; `todo` prepara, ejecuta los modelos, evalúa y grafica. Los ajustes ya terminados se reutilizan. Para ejecutar solo una familia se puede usar `naive`, `sarima` o `lstm` en lugar de `todo`.

Para recalcular tablas y figuras desde las predicciones guardadas, **sin entrenar**:

```powershell
.\.venv\Scripts\python.exe run.py evaluar --salida resultados/reproduccion
.\.venv\Scripts\python.exe run.py figuras --salida resultados/reproduccion
```

La carpeta elegida debe contener las predicciones de esa ejecución. Omitir `--salida` usa `resultados/` y vuelve a escribir las tablas o figuras definitivas. Para experimentar, usar otra carpeta. `--estacion` y `--objetivo` limitan las etapas de modelos a un caso; `evaluar` y `figuras` resumen toda la carpeta elegida. `run.py --help` muestra estas opciones.

## Protocolo que debe mantenerse

| Bloque | Inicio incluido | Fin excluido | Horas |
|---|---|---|---:|
| Train | 2013-03-01 | 2015-12-19 | 24552 |
| Validation | 2015-12-19 | 2016-07-25 | 5256 |
| Test | 2016-07-25 | 2017-03-01 | 5256 |

Los bloques abarcan aproximadamente 70/15/15 % del período. Los primeros siete días son contexto: quedan 1016 ejemplos diarios de Train y 219 por bloque de evaluación. El **día objetivo** determina el bloque. Solo se utilizan mediciones disponibles hasta el origen.

Las entradas faltantes usan el último valor conocido (`ffill`); sin pasado, cero y máscara de ausencia. La normalización usa exclusivamente Train observado. Los objetivos faltantes se excluyen del error; los ceros originales sí se evalúan. No se recortan extremos ni predicciones negativas.

LSTM: historia de 48 horas, una capa de 16 unidades en cada componente y 24 salidas autoregresivas. `base` usa objetivo, máscara y calendario; `meteo` añade cinco variables meteorológicas históricas y sus máscaras. El Decoder recibe solo el valor anterior y calendario futuro, sin Teacher Forcing. Adam 0.003, batch máximo 256, hasta 25 épocas, paciencia 4 y clipping de norma global 1.

Se entrena con **MSE normalizado enmascarado**, se recupera la época por **MSE de Validation** y se elige la variante por **MAE de Validation en µg/m³**, con semilla 2026. La principal se repite con 2027 y 2028. Test evalúa decisiones ya fijadas. SARIMA usa `(1,0,0)(1,0,0,24)` con constante; sus diagnósticos se guardan en `resultados/modelos/*_sarima.json`. El diagnóstico Ljung–Box es descriptivo: al omitir residuos faltantes se comprimen los rezagos.

## Resultados y cumplimiento

| Requisito | Evidencia en `resultados/` |
|---|---|
| 6 casos, 12 referencias Naive, 6 ajustes SARIMA y 24 entrenamientos LSTM | `metricas/inventario.json` y `modelos/` |
| Dos variantes y tres semillas de la principal | `modelos/*_seleccion.json` y `metricas/variabilidad.csv` |
| MAE, RMSE y cobertura en Validation y Test | `metricas/comparacion.csv` |
| Origen, hora objetivo, horizontes 1–24, observación y estimación | `predicciones/*.csv.gz` |
| Errores por horizonte, trayectorias y aprendizaje | `metricas/por_horizonte.csv`, curvas y `figuras/` |
| Preparación, configuración, tiempos y hardware | `modelos/*_preparacion*`, JSON de ajustes, `configuracion.json` y `entorno.json` |

Hay 441504 filas de predicciones. Validation seleccionó `base` en los tres casos PM2.5 y `meteo` en los tres O3. Se conservan todas las semillas. `codigo_sha256.json` respalda el código de la ejecución original, anterior a las simplificaciones.

Como apoyo opcional, `exploracion.ipynb`, ubicado en la raíz, conserva la exploración ejecutada de Train. Las explicaciones de las operaciones y dimensiones están en los comentarios de los módulos Python. Para revisar los experimentos basta abrir los CSV y las figuras. Las comprobaciones críticas del código usan `assert`.

El informe PDF y la presentación PPTX con copia PDF son entregables requeridos, pero actualmente no están incluidos en este repositorio. **Su preparación, identificación del grupo, interpretación y conclusión definitivas quedan pendientes a cargo de los estudiantes.**
