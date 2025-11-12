import dotenv
import os
import uvicorn
from fastapi import FastAPI
from routers import chatbot

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print('Error|dotenv module not found. Please install it using "pip install python-dotenv".')

app = FastAPI(
    title="VibeThinker API",
    description="API para interactuar con el modelo VibeThinker.",
    version="1.0.0"
)

app.include_router(chatbot.router)

if __name__ == '__main__':
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)