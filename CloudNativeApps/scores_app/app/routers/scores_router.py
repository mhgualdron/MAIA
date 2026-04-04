from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import ScoreCreate, ScoreResponse
from app.services.scores_service import ScoreService
from uuid import UUID

router = APIRouter(prefix="/scores", tags=["scores"])

def get_score_service(db: Session = Depends(get_db)) -> ScoreService:
    return ScoreService(db)

@router.get("/ping")
def ping():
    return "pong"

@router.post("/reset")
def reset(service: ScoreService = Depends(get_score_service)):
    service.reset()
    return {"msg": "Todos los datos fueron eliminados"}

@router.post("", status_code=status.HTTP_201_CREATED, response_model=ScoreResponse)
def create_score(payload: ScoreCreate, service: ScoreService = Depends(get_score_service)):
    return service.create(payload)

@router.get("/count")
def count(service: ScoreService = Depends(get_score_service)):
    return {"count": service.count()}

@router.get("/{offer_id}")
def get_score(offer_id: str, service: ScoreService = Depends(get_score_service)):
    try:
        UUID(offer_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Identificador inválido")

    score = service.get_by_offer(offer_id)
    if not score:
        raise HTTPException(status_code=404, detail="Score no encontrado")
    return score
