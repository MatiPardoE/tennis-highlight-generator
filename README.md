# Tennis Highlight Generator (MVP)

Aplicación local en Streamlit para recortar videos de tenis amateur, eliminando tiempos muertos y conservando segmentos de juego real con una heurística simple basada en detección de jugadores + optical flow local.

## Stack

- Python 3.11+
- Streamlit
- OpenCV
- NumPy
- Ultralytics YOLO
- FFmpeg (CLI en `PATH`)
- `pathlib`, `dataclasses`, `typing`

## Estructura

```text
.
├── app.py
├── requirements.txt
├── README.md
└── src
    ├── __init__.py
    ├── config.py
    ├── models.py
    ├── video_io.py
    ├── motion_features.py
    ├── debug_export.py
    ├── segment_detection.py
    ├── postprocess.py
    ├── export.py
    ├── preview.py
    └── utils.py
```

## Cómo funciona

1. Se muestrea el video a una tasa configurable (`sample_fps`).
2. Se detectan personas con YOLO y se filtra solo clase `person`.
3. En cada frame se eligen los 2 jugadores principales por área de bounding box.
4. Se calcula optical flow Farneback entre frames consecutivos.
5. Se calcula score local dentro de cada bounding box y se combina en un score global.
6. Se suaviza la señal temporal.
7. Se aplica umbral dinámico (percentiles + sensibilidad).
8. Se generan segmentos GAME iniciales.
9. Postproceso temporal:
   - merge de segmentos cercanos
   - eliminación de segmentos muy cortos
   - padding antes/después
10. Se exporta video final con FFmpeg concatenando solo segmentos GAME.

## Instalación

1. Crear entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell
```

2. Instalar dependencias Python:

```bash
pip install -r requirements.txt
```

3. Verificar FFmpeg:

```bash
ffmpeg -version
ffprobe -version
```

## Ejecución

```bash
streamlit run app.py
```

Abrí la URL local que muestra Streamlit (normalmente `http://localhost:8501`).

## Uso

- Subir video (`.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`)
- Ajustar parámetros:
  - Sensibilidad
  - Duración mínima de juego
  - Duración mínima de pausa
  - Padding antes/después
  - Muestreo de frames
  - Suavizado temporal
  - Modelo YOLO + confianza mínima de persona
  - Métrica local de flow (`p90`, `mean`, `fast_ratio`)
  - Umbral de flow rápido y modo de combinación global (`max` o `mean`)
  - Modo debug (video con bounding boxes y scores de actividad)
- Procesar video
- Ver resumen de segmentos detectados
- En debug: ver y descargar un video con cajas de jugadores y score por frame
- Ver profiling por etapa (tiempos de pipeline) y métricas del extractor de movimiento
- Descargar `pipeline.log` y `profiling.json` de cada corrida
- Visualizar export final
- Descargar highlights

## Profiling y logs

- Cada ejecución crea un `workspace` temporal.
- En ese workspace se guardan:
  - `pipeline.log`: eventos y tiempos de cada etapa.
  - `profiling.json`: resumen de tiempos por etapa y métricas de extracción (`YOLO`, `flow`, throughput).
- La UI muestra una tabla de tiempos por etapa y botones para descargar ambos archivos.

## Notas y límites del MVP

- No hay detección de pelota ni tracking persistente de jugadores.
- Cambios bruscos de cámara y detecciones inestables pueden afectar la precisión.
- Parámetros pueden necesitar ajuste según cada video.
- Se prioriza mantenibilidad y rapidez de iteración.

## Próximas mejoras sugeridas

- Detección de región de cancha para reducir ruido de fondo.
- Umbral adaptativo por tramos del video.
- Cache de features para reintentos rápidos con distintos parámetros.
- Export opcional con overlays de segmentos para auditoría.
- Tests unitarios para postproceso y conversión de máscaras a segmentos.
