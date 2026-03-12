# Tennis Highlight Generator (MVP)

Aplicación local en Streamlit para recortar videos de tenis amateur, eliminando tiempos muertos y conservando segmentos de juego real usando heurísticas simples de movimiento (sin modelos entrenados).

## Stack

- Python 3.11+
- Streamlit
- OpenCV
- NumPy
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
    ├── segment_detection.py
    ├── postprocess.py
    ├── export.py
    ├── preview.py
    └── utils.py
```

## Cómo funciona

1. Se muestrea el video a una tasa configurable (`sample_fps`).
2. Se calcula actividad por diferencia absoluta entre frames consecutivos en escala de grises.
3. Se suaviza la señal temporal.
4. Se aplica umbral dinámico (percentiles + sensibilidad).
5. Se generan segmentos GAME iniciales.
6. Postproceso temporal:
   - merge de segmentos cercanos
   - eliminación de segmentos muy cortos
   - padding antes/después
7. Se exporta video final con FFmpeg concatenando solo segmentos GAME.

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
  - Modo debug (incluye video de máscara de movimiento por umbral)
- Procesar video
- Ver resumen de segmentos detectados
- En debug: ver y descargar un video donde se iluminan solo los píxeles con diferencia que supera el umbral
- Visualizar export final
- Descargar highlights

## Notas y límites del MVP

- No hay detección de pelota ni modelos ML.
- Cambios bruscos de cámara/fondo pueden afectar la precisión.
- Parámetros pueden necesitar ajuste según cada video.
- Se prioriza mantenibilidad y rapidez de iteración.

## Próximas mejoras sugeridas

- Detección de región de cancha para reducir ruido de fondo.
- Umbral adaptativo por tramos del video.
- Cache de features para reintentos rápidos con distintos parámetros.
- Export opcional con overlays de segmentos para auditoría.
- Tests unitarios para postproceso y conversión de máscaras a segmentos.
