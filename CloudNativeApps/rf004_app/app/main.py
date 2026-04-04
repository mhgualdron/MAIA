from fastapi import FastAPI, Request, Header, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.schemas import OfferCreateRequest, OfferResponse
from app.services.orchestrator import OrchestratorService
from typing import Optional

app = FastAPI(title="RF-004 Orchestrator", version="1.0.0")
service = OrchestratorService()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"msg": "Bad Request"},
        headers={"X-Orchestrator-Version": "V2-Resilient"}
    )

@app.get("/rf004/ping")
@app.get("/ping")
async def ping():
    return JSONResponse(
        content="pong",
        headers={"X-Orchestrator-Version": "V2-Resilient"}
    )

@app.post("/rf004/posts/{post_id}/offers", status_code=201, response_model=OfferResponse)
async def create_offer_rf004(
    post_id: str,
    request: OfferCreateRequest,
    authorization: Optional[str] = Header(None)
):
    # Project Rule: Token must be in "Bearer <token>" format
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(
            status_code=403,
            content={"msg": "Token no está en la solicitud"},
            headers={"X-Orchestrator-Version": "V2-Resilient"}
        )
    
    token = authorization.split(" ")[1]
    
    try:
        # This call now uses the shielded/resilient logic
        result = await service.process_rf004(post_id, token, request)
        return result
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"msg": e.detail},
            headers={"X-Orchestrator-Version": "V2-Resilient"}
        )
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"msg": "El servicio está temporalmente fuera de servicio."},
            headers={"X-Orchestrator-Version": "V2-Resilient"}
        )
