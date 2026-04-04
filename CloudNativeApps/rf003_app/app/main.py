from fastapi import FastAPI, Request, Header, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.schemas import PostCreateRequest, PostResponse
from app.services.orchestrator import OrchestratorService
from typing import Optional

app = FastAPI(title="RF-003 Orchestrator", version="1.0.0")
service = OrchestratorService()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"msg": "Bad Request"},
        headers={"X-Orchestrator-Version": "V2-Resilient"}
    )

@app.get("/rf003/ping")
@app.get("/ping")
@app.get("/rf003/health")
@app.get("/health")
async def ping():
    return JSONResponse(
        content="pong",
        headers={"X-Orchestrator-Version": "V3-ReadinessFix"}
    )

@app.post("/rf003/posts", status_code=201, response_model=PostResponse)
async def create_post_rf003(
    request: PostCreateRequest,
    authorization: Optional[str] = Header(None)
):
    if not all([request.flightId, request.expireAt, request.plannedStartDate, request.plannedEndDate, request.origin, request.destiny, request.bagCost is not None]):
        return JSONResponse(
            status_code=400,
            content={"msg": "Faltan campos en la solicitud"},
            headers={"X-Orchestrator-Version": "V2-Resilient"}
        )

    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(
            status_code=403,
            content={"msg": "Token no está en la solicitud"},
            headers={"X-Orchestrator-Version": "V2-Resilient"}
        )
    
    token = authorization.split(" ")[1]
    
    try:
        result = await service.process_rf003(token, request)
        return result
    except HTTPException as e:
        print(f"RESILIENCE-DEBUG: Caught HTTPException: {e.status_code} - {e.detail}")
        return JSONResponse(
            status_code=e.status_code,
            content={"msg": e.detail},
            headers={"X-Orchestrator-Version": "V4-ExplicitStatus"}
        )
    except Exception as e:
        print(f"RESILIENCE-DEBUG: Caught unexpected Exception: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"msg": "El servicio está temporalmente fuera de servicio."},
            headers={"X-Orchestrator-Version": "V4-ExplicitStatus"}
        )
