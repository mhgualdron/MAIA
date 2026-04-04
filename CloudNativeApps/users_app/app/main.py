from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.database import engine, Base
from app.routers import user_router
from contextlib import asynccontextmanager



# Lifespan para crear la base de datos
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield

# Creación de la aplicación
app = FastAPI(title="Users Microservice", lifespan=lifespan, version="1.0.0")

# Exception handler para convertir 422 a 400
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    import traceback
    print(f"DEBUG: Validation error for {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()},
    )

# Inclusión de routers
app.include_router(user_router.router)
