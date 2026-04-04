from fastapi import FastAPI, Request
from fastapi.responses import Response
from app.routers import posts
from app.database import engine, Base
from contextlib import asynccontextmanager
from fastapi.exceptions import RequestValidationError

# Context manager para manejar startup y shutdown de la app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: crear las tablas de la base de datos si no existen
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Aquí se podrían agregar tareas de shutdown si fueran necesarias

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="Posts App",
    version="1.0.0",
    lifespan=lifespan
)

# Handler personalizado para errores de validación de requests
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Responde con status code 400 sin cuerpo
    return Response(status_code=400)

# Registro de routers
app.include_router(posts.router)