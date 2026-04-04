# 🎵 AudioAPP — Clasificación de Audio con FastAPI

Microservicio FastAPI para clasificación de audio usando modelos de Deep Learning entrenados en PyTorch.

## Descripción

API REST que recibe archivos de audio y retorna una predicción de clase usando un modelo pre-entrenado de clasificación (e.g. ESC-50 dataset).

## Archivos

| Archivo | Descripción |
|---------|-------------|
| [`main.py`](./main.py) | Aplicación FastAPI principal |
| [`requirements.txt`](./requirements.txt) | Dependencias Python |

## 🚀 Cómo ejecutar

```bash
# Instalar dependencias
pip install -r requirements.txt

# Levantar el servidor
uvicorn main:app --reload --port 8000

# Probar la API
curl -X POST "http://localhost:8000/predict" \
  -F "file=@audio_sample.wav"
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Clasificar archivo de audio |

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **ML**: PyTorch / librosa
- **Audio processing**: librosa, soundfile
