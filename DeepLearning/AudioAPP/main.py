import torch
import librosa
import numpy as np
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoModelForAudioClassification, ASTFeatureExtractor

# --- CONFIGURACIÓN ---
MODEL_PATH = "ast_esc50_production" 

app = FastAPI(title="AST Audio Classifier API")

# Variables globales para cargar el modelo una sola vez al inicio
model = None
feature_extractor = None
id2label = {}

@app.on_event("startup")
def load_model():
    global model, feature_extractor, id2label
    print("Cargando modelo AST... (esto puede tardar un poco)")
    
    # Cargar Feature Extractor
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_PATH)
    
    # Cargar Modelo
    model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)
    model.eval() # Poner en modo evaluación (apaga dropout, etc)
    
    # Recuperar el mapa de etiquetas desde la config del modelo
    id2label = model.config.id2label
    print("Modelo cargado exitosamente.")

def preprocess_audio(audio_bytes):
    """
    Lee bytes de MP3, resamplea a 16kHz y prepara tensores.
    """
    # 1. Cargar audio desde memoria usando librosa
    # target_sr=16000 es MANDATORIO para AST
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    
    # 2. Asegurar duración (padding o recorte si es necesario, el extractor lo maneja)
    # AST suele esperar clips de ~10 segundos, pero el extractor hace padding automático
    
    # 3. Feature Extraction
    inputs = feature_extractor(
        y, 
        sampling_rate=16000, 
        padding="max_length", 
        return_tensors="pt"
    )
    return inputs

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    # Validar formato
    if not file.filename.endswith(('.mp3', '.wav', '.ogg')):
        raise HTTPException(status_code=400, detail="Formato no soportado. Usa MP3 o WAV.")
    
    try:
        # Leer archivo
        contents = await file.read()
        
        # Preprocesar
        inputs = preprocess_audio(contents)
        
        # Inferencia
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Post-procesamiento (Logits -> Probabilidades -> Clase)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_id = torch.argmax(logits, dim=-1).item()
        score = probabilities[0][predicted_id].item()
        
        label_name = id2label[predicted_id]
        
        return {
            "filename": file.filename,
            "prediction": label_name,
            "confidence": round(score, 4),
            "label_id": predicted_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)