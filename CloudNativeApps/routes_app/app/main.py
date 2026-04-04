from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.routers import routes
from app.database import engine, Base
import asyncio

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Shutdown logic if needed

app = FastAPI(title="Routes App", version="1.0.0", lifespan=lifespan)

from fastapi.exceptions import RequestValidationError
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors(), "body": exc.body},
    )

app.include_router(routes.router)
