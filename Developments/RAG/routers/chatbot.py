from fastapi import APIRouter
from pydantic import BaseModel
import os
from utils.utils import VibeThinker

router = APIRouter()

# Cargar el modelo una vez al iniciar el servidor
model_path = os.getenv("MODEL_PATH", "WeiboAI/VibeThinker-1.5B")
model = VibeThinker(model_path)

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

@router.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def infer(request: ChatRequest):
    """Recibe un prompt y devuelve la inferencia del modelo."""
    model_response = model.infer_text(request.prompt)
    return {"response": model_response}