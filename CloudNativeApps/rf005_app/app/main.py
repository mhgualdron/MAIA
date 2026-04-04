from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.routers.rf005 import router
from starlette.exceptions import HTTPException as StarletteHTTPException

app = FastAPI(title="RF-005 Orchestrator", version="1.0.0")

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"msg": exc.detail} if hasattr(exc, "detail") else {"msg": "Error"}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"msg": "Bad Request"}
    )

@app.get("/ping")
@app.get("/rf005/ping")
async def ping():
    return JSONResponse(content="pong")

app.include_router(router)
