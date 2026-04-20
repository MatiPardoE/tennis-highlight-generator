# Tennis Highlight Generator

Aplicación local en Python para analizar videos de tenis sin subir archivos pesados a la nube.

La arquitectura está separada en tres etapas:

1. `preprocess.py`: genera un proxy local liviano, opcionalmente recortado desde un inicio útil, manteniendo el mapeo temporal al original.
2. `detect_segments.py`: analiza el proxy y detecta segmentos útiles en segundos del original.
3. `render_from_segments.py`: corta y concatena desde el video original en máxima calidad.

El manifiesto JSON resultante queda listo para auditar, reusar o volver a renderizar sin repetir el análisis.

## Principios del diseño

- Todo corre en local.
- El análisis se hace sobre un proxy optimizado.
- El render final se hace siempre sobre el original.
- Los timestamps del proxy mapean 1:1 al original porque no hay cambios de velocidad ni re-timing.
- El frame elegido para calibración puede usarse también como inicio útil del proxy para saltar la parte donde todavía se acomoda la cámara.
- Rotación libre en grados y ROI poligonal de 6 puntos se fijan antes del análisis y se aplican al proxy.
- La detección usa YOLO para ubicar jugadores y optical flow local dentro de sus cajas.

## Archivos principales

```text
.
├── app.py
├── main.py
├── config.py
├── preprocess.py
├── detect_segments.py
├── render_from_segments.py
└── requirements.txt
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ffmpeg -version
ffprobe -version
```

## Uso por CLI

### 1. Generar proxy

```bash
python main.py preprocess \
  --input /videos/partido.mp4 \
  --workspace ./runs/partido \
  --analysis-start-sec 3.0 \
  --rotation 2.5 \
  --roi "180,120;900,90;1700,120;1580,920;960,970;340,920" \
  --proxy-fps 8 \
  --proxy-width 640
```

### 2. Detectar segmentos

```bash
python main.py detect \
  --manifest ./runs/partido/segments_manifest.json \
  --sensitivity 0.55 \
  --smooth-window-sec 1.0 \
  --min-segment-sec 4.0 \
  --min-gap-sec 2.0 \
  --padding-sec 0.35 \
  --yolo-model yolov8n.pt \
  --person-confidence 0.30
```

### 3. Renderizar desde el original

```bash
python main.py render \
  --manifest ./runs/partido/segments_manifest.json \
  --output ./runs/partido/highlights.mp4 \
  --crf 18 \
  --preset medium
```

### Pipeline completo

```bash
python main.py run-all \
  --input /videos/partido.mp4 \
  --workspace ./runs/partido \
  --analysis-start-sec 3.0 \
  --rotation 2.5 \
  --roi "180,120;900,90;1700,120;1580,920;960,970;340,920" \
  --proxy-fps 8 \
  --proxy-width 640 \
  --sensitivity 0.55 \
  --min-segment-sec 4.0 \
  --min-gap-sec 2.0 \
  --padding-sec 0.35 \
  --output ./runs/partido/highlights.mp4
```

## Uso con Streamlit

La UI ahora trabaja con paths locales, no con uploads:

```bash
streamlit run app.py
```

Permite:

- indicar path del video original
- elegir ángulo libre de rotación
- elegir el frame de calibración por tiempo; ese mismo instante pasa a ser el inicio útil del proxy
- definir un ROI poligonal de 6 puntos con preview en vivo
- ajustar parámetros de YOLO y optical flow
- generar proxy
- detectar segmentos
- guardar un video debug de detección dentro del `run`
- renderizar highlights
- inspeccionar el JSON final

## Defaults recomendados

- `rotation=0.0` salvo que el video necesite corrección fina
- `roi=` polígono de 6 puntos alrededor de la cancha útil
- `proxy_fps=8`
- `proxy_width=640`
- `proxy_crf=34`
- `sensitivity=0.55`
- `smooth_window_sec=1.0`
- `min_segment_sec=4.0`
- `min_gap_sec=2.0`
- `padding_sec=0.35`
- `yolo_model=yolov8n.pt`
- `person_confidence=0.30`
- `render_crf=18`
- `render_preset=medium`

## Formato del JSON

Ejemplo:

```json
{
  "original_video_path": "/videos/partido.mp4",
  "proxy_video_path": "/runs/partido/proxy.mp4",
  "rotation": 2.5,
  "analysis_start_sec": 3.0,
  "roi": {
    "points": [
      { "x": 180, "y": 120 },
      { "x": 900, "y": 90 },
      { "x": 1700, "y": 120 },
      { "x": 1580, "y": 920 },
      { "x": 960, "y": 970 },
      { "x": 340, "y": 920 }
    ]
  },
  "fps_original": 29.97,
  "fps_proxy": 8.0,
  "useful_segments": [
    {
      "start_sec": 15.2,
      "end_sec": 41.7
    },
    {
      "start_sec": 58.4,
      "end_sec": 79.1
    }
  ],
  "original_width": 1920,
  "original_height": 1080,
  "proxy_width": 640,
  "proxy_height": 362,
  "duration_original_sec": 3600.0,
  "duration_proxy_sec": 3597.0,
  "debug_video_path": "/runs/partido/debug_detection.mp4",
  "roi_mask_path": "/runs/partido/roi_mask.png"
}
```

## Decisiones técnicas

- `ffmpeg` para generar el proxy: es más robusto y eficiente para videos grandes que rehacer el encode frame a frame con OpenCV.
- Rotación + autocrop + crop al bounding box del polígono + máscara poligonal + grayscale + scale + fps en una sola pasada: menos I/O y sin bordes negros en ángulos no enteros.
- El inicio útil se aplica en el proxy y luego la detección vuelve a sumar ese offset para guardar segmentos en segundos del original.
- El proxy sigue siendo un transcode completo; para acelerarlo se prioriza `ultrafast`, CRF alto, `hwaccel auto` y progreso visible desde `ffmpeg`.
- La detección aplica YOLO sobre el proxy y calcula optical flow local dentro de las cajas de los jugadores detectados.
- El render final aplica la misma geometría calibrada del proxy, incluida la máscara poligonal, para que los highlights salgan con la rotación y recorte definidos en calibración.
- Detección por diferencia entre frames en el proxy: primera versión mantenible y barata computacionalmente.
- JSON único de manifiesto: evita recalcular si solo querés re-renderizar.
- Render sobre original con `trim` + `concat`: garantiza calidad final y mantiene independencia entre análisis y export.

## Comando ffmpeg usado para exportar desde el original

La implementación usa un `filter_complex` equivalente a este patrón:

```bash
ffmpeg -y -i original.mp4 \
  -filter_complex "[0:v]trim=start=15.2:end=41.7,setpts=PTS-STARTPTS[v0]; \
                   [0:v]trim=start=58.4:end=79.1,setpts=PTS-STARTPTS[v1]; \
                   [v0][v1]concat=n=2:v=1:a=0[vout]" \
  -map "[vout]" \
  -c:v libx264 -preset medium -crf 18 \
  highlights.mp4
```

Si hay audio y `ffprobe` está disponible, el pipeline concatena audio y video.
