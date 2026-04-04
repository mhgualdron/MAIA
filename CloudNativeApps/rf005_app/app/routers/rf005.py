from fastapi import APIRouter, Header, Depends, HTTPException
from app.schemas import RF005Response
from app.services.orchestrator import OrchestratorService

router = APIRouter(tags=["rf005"])

async def get_orchestrator():
    service = OrchestratorService()
    try:
        yield service
    finally:
        await service.close()

@router.head("/rf005/posts/{post_id}", status_code=200)
async def check_availability(post_id: str):
    return {"status": "ok"}

@router.get("/rf005/posts/{post_id}", response_model=RF005Response, status_code=200)
async def get_post_details_orchestrated(
    post_id: str,
    authorization: str = Header(default=None),
    service: OrchestratorService = Depends(get_orchestrator)
):
    if not authorization:
        raise HTTPException(
            status_code=403,
            detail="No está autorizado para realizar esta acción."
        )
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="El token no es válido o está vencido."
        )

    token = authorization.replace("Bearer ", "").strip()
    
    return await service.process_rf005(post_id, token)
