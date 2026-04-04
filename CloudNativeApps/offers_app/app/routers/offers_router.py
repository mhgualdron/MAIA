from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import OfferCreate, OfferResponse
from app.services.offers_service import OfferService
from uuid import UUID

router = APIRouter(prefix="/offers", tags=["offers"])

def get_offer_service(db: Session = Depends(get_db)) -> OfferService:
    return OfferService(db)

@router.get("/ping")
def ping():
    return "pong"

@router.post("/reset")
def reset(service: OfferService = Depends(get_offer_service)):
    service.reset()
    return {"msg": "Todos los datos fueron eliminados"}

@router.post("", status_code=status.HTTP_201_CREATED, response_model=OfferResponse)
def create_offer(payload: OfferCreate, service: OfferService = Depends(get_offer_service)):
    if payload.offer < 0:
         raise HTTPException(status_code=412, detail="La oferta no puede ser negativa")
    
    if payload.size not in ["LARGE", "MEDIUM", "SMALL"]:
         raise HTTPException(status_code=412, detail="Tamaño inválido")

    return service.create(payload)

@router.get("/count")
def count(service: OfferService = Depends(get_offer_service)):
    return {"count": service.count()}

@router.get("")
def list_offers(
    post: str | None = Query(None),
    owner: str | None = Query(None),
    service: OfferService = Depends(get_offer_service)
):
    return service.list_offers(postId=post, userId=owner)

@router.get("/{id}")
def get_offer(id: str, service: OfferService = Depends(get_offer_service)):
    try:
        UUID(id)
    except Exception:
        raise HTTPException(status_code=400, detail="Identificador inválido")

    offer = service.get_offer(id)
    if not offer:
        raise HTTPException(status_code=404, detail="Oferta no encontrada")
    return offer

@router.delete("/{id}")
def delete_offer(id: str, service: OfferService = Depends(get_offer_service)):
    try:
        UUID(id)
    except Exception:
        raise HTTPException(status_code=400, detail="Identificador inválido")
    
    if service.delete_offer(id):
        return {"msg": "la oferta fue eliminada"}
    
    raise HTTPException(status_code=404, detail="Oferta no encontrada")
