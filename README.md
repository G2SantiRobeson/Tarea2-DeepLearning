# Tarea 2 · Pronóstico de calidad del aire en Beijing

Se comparan **Naive diario y semanal, SARIMA y LSTM Seq2Seq** para PM2.5 y O3 en Aotizhongxin, Changping y Dingling: seis casos. A las **23:00 del reloj nominal del CSV** se pronostican las 24 horas del día siguiente. Los resultados incluidos corresponden a **máximo 50 épocas, Early Stopping con paciencia 4 y semillas 2026, 2027 y 2028**.

## Estructura

| Ruta | Contenido |
|---|---|
| `src/datos.py` | Lectura, preparación causal y ventanas |
| `src/modelos_naive.py`, `src/modelo_sarima.py` | Referencias y modelo estadístico |
| `src/modelo_lstm.py`, `src/entrenamiento.py` | Encoder/Decoder, entrenamiento e inferencia |
| `src/evaluacion.py`, `src/graficos.py` | Métricas comparables y figuras |
| `config.py`, `run.py` | Configuración única y ejecución por etapas |
| `data/` | `beijing_consolidated.csv` y `diccionario.csv` originales |
| `exploracion.ipynb` | Exploración de Train y consulta de resultados guardados |
| `resultados/modelos/` | Checkpoints `.pt`, ajustes SARIMA `.pkl`, preparación, selección y tiempos |
| `resultados/predicciones/` | 84 CSV comprimidos de Validation y Test |
| `resultados/metricas/` | Comparación, errores por horizonte, variabilidad, curvas e inventario |
| `resultados/figuras/` | Figuras por caso y exploración de Train |
| `resultados/figuras/informe/` | Seis figuras compuestas y sus seis generadores Python |
| `resultados/configuracion.json`, `resultados/entorno.json` | Protocolo, SHA-256 del dataset y entorno registrado |

Los datos y artefactos necesarios están incluidos. La ejecución no depende de informes, enunciados, respaldos ni carpetas externas.

## Instalación y datos

Se usó Python 3.12.10 en CPU. Desde la raíz del repositorio, en PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py --help
```

Si `.venv` ya existe, reutilizarlo. En Linux/macOS, usar `python3` y `.venv/bin/python`. Las versiones registradas están fijadas en `requirements.txt`; no se requieren herramientas de generación de documentos. El notebook puede consultarse con sus salidas guardadas; para ejecutarlo se necesita además un frontend Jupyter, no incluido en las dependencias del flujo de consola.

Conservar ambos CSV en `data/`, sin imputarlos ni normalizarlos manualmente. El consolidado contiene una grilla horaria por estación y banderas de observación; `diccionario.csv` describe columnas, unidades y procedencia. `src/datos.py` valida unicidad, grilla y máscaras, y prepara cada caso al ejecutarlo. El hash de los datos debe coincidir con `resultados/configuracion.json`.

## Reproducir en una carpeta nueva

Los resultados definitivos ya están en `resultados/`. Para reproducir los seis casos sin sobrescribirlos, crear un nombre de salida único:

```powershell
$salida = Join-Path (Resolve-Path ..) ('reproducciones/experimento_' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe run.py todo --salida $salida
```

Para un solo caso, añadir `--estacion Aotizhongxin --objetivo pm25`. `todo` prepara datos, ejecuta modelos, evalúa y grafica. También se puede elegir `auditoria`, `naive`, `sarima` o `lstm`. `auditoria` guarda preparación y cobertura sin entrenar. La ejecución completa en una salida nueva realiza seis ajustes SARIMA y 24 entrenamientos LSTM; las referencias Naive no requieren ajuste.

Al reutilizar una salida, los ajustes existentes pueden cargarse, pero algunas predicciones, tiempos, tablas y figuras vuelven a escribirse. **Omitir `--salida` usa la carpeta oficial `resultados/`.** Usar siempre otra carpeta para reproducir o experimentar. `--estacion` y `--objetivo` limitan las etapas de modelos; `evaluar` y `figuras` procesan toda la carpeta elegida.

Para verificar tablas y regenerar figuras **sin entrenar**, trabajar sobre una copia nueva de los resultados:

```powershell
$revision = Join-Path (Resolve-Path ..) ('reproducciones/revision_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $revision | Out-Null
Copy-Item -Path .\resultados\* -Destination $revision -Recurse
.\.venv\Scripts\python.exe run.py figuras --salida $revision
Get-ChildItem (Join-Path $revision 'figuras/informe/generar_*.py') | ForEach-Object { & .\.venv\Scripts\python.exe $_.FullName }
```

`figuras` evalúa las predicciones existentes y genera las figuras ordinarias. Para generar solo tablas, usar `evaluar`. Los seis generadores compuestos leen las fuentes relativas a su propia ubicación y escriben PNG a 300 DPI junto al script: ejecutar las copias, como en el ejemplo. Las tres figuras de exploración de Train se conservan desde el notebook.

## Protocolo experimental

| Bloque | Inicio incluido | Fin excluido | Horas |
|---|---|---|---:|
| Train | 2013-03-01 | 2015-12-19 | 24552 |
| Validation | 2015-12-19 | 2016-07-25 | 5256 |
| Test | 2016-07-25 | 2017-03-01 | 5256 |

Los bloques abarcan aproximadamente 70/15/15 % del período. Los primeros siete días son contexto: quedan 1016 ejemplos diarios de Train y 219 por bloque de evaluación. El **día objetivo** determina el bloque. Solo se utilizan mediciones disponibles hasta el origen.

Las entradas faltantes usan el último valor conocido (`ffill`); sin pasado, cero y máscara de ausencia. La normalización se ajusta exclusivamente con Train observado. Los objetivos faltantes se excluyen del error; los ceros originales sí se evalúan. No se recortan extremos ni predicciones negativas.

LSTM: historia de 48 horas, una capa de 16 unidades en cada componente y 24 salidas autoregresivas. `base` usa objetivo, máscara y calendario; `meteo` añade cinco variables meteorológicas históricas y sus máscaras. El Decoder recibe el valor anterior y calendario futuro, sin Teacher Forcing. Adam 0.003, batch máximo 256, **máximo 50 épocas**, paciencia 4, clipping de norma global 1 y dos hilos. Los 24 entrenamientos terminaron por Early Stopping entre **5 y 47 épocas**; 50 es el máximo permitido.

La pérdida es **MSE normalizado enmascarado**. Se recupera la época de menor MSE de Validation normalizado. La variante principal se elige por **MAE de Validation en escala original (µg/m³)**, comparando ambas variantes con semilla 2026; solo la principal se repite con 2027 y 2028. Test evalúa decisiones ya fijadas.

| Estación | Variante PM2.5 | Variante O3 |
|---|---|---|
| Aotizhongxin | Base | Meteorológica |
| Changping | Base | Meteorológica |
| Dingling | Meteorológica | Meteorológica |

Cada caso conserva cuatro entrenamientos: ambas variantes con 2026 y la principal con 2027 y 2028. En Dingling–PM2.5 las repeticiones son `meteo_2027` y `meteo_2028`.

SARIMA usa `(1,0,0)(1,0,0,24)` con constante; sus diagnósticos están en `resultados/modelos/*_sarima.json`. Ljung–Box es descriptivo: al omitir residuos faltantes se comprimen los rezagos.

## Resultados y reproducibilidad

`metricas/inventario.json` registra seis casos, 12 referencias Naive, seis ajustes SARIMA y 24 LSTM. Los 84 CSV de predicciones contienen 441504 filas con origen, hora objetivo, horizonte, observación, máscara y estimación. `comparacion.csv`, `por_horizonte.csv` y `variabilidad.csv` conservan métricas, cobertura y semillas; `modelos/*_seleccion.json` documenta la selección por Validation. Las cifras originales se consultan directamente en estos archivos.

Los checkpoints contienen configuración, preparación, arquitectura, semilla, pesos y mejor época, lo que permite comprobar compatibilidad e inferencia. **No se dispone de una revisión o manifiesto del código usado en el entrenamiento que demuestre su identidad histórica exacta.** Verificar artefactos e inferencia no equivale a repetir el entrenamiento ni certifica esa procedencia. El notebook conserva sus salidas de exploración y consulta; los contadores nulos de las consultas finales no acreditan una ejecución completa de todas las celdas.
