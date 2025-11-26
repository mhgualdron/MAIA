# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from chat import process_user_message # Necesitarás crear esta función
import uvicorn
import os

app = FastAPI(title="LangGraph Agent API")

class MessageInput(BaseModel):
    user_input: str
    thread_id: str = "1"

@app.post("/chat")
def chat_endpoint(input: MessageInput):
    """
    Recibe un mensaje de usuario, lo procesa con el agente de LangGraph, 
    y devuelve la respuesta del agente.
    """
    try:
        agent_response = process_user_message(
            user_input=input.user_input,
            thread_id=input.thread_id
        )
        return {"response": agent_response}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)