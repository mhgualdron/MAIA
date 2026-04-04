from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.core.database import get_db
from src.domain.credit_card import CreditCard
from src.api.schemas import CreditCardCreate, CreditCardResponse, CreditCardDetail, CountResponse, ResetResponse
from src.api.dependencies import auth_user
from src.services.truenative_service import register_card
from src.services.sqs_publisher import publish_card_event
import datetime
import uuid

router = APIRouter(prefix="/credit-cards", tags=["credit-cards"])

def is_expired(expiration_date: str) -> bool:
    try:
        if "/" not in expiration_date:
            return True
        year_str, month_str = expiration_date.split("/")
        month = int(month_str)
        year = int(year_str)
        if not (1 <= month <= 12):
            return True
        full_year = 2000 + year
        if month == 12:
            next_month = datetime.date(full_year + 1, 1, 1)
        else:
            next_month = datetime.date(full_year, month + 1, 1)
        exp_date_last_day = next_month - datetime.timedelta(days=1)
        if exp_date_last_day < datetime.date.today():
            return True
        return False
    except Exception:
        return True

@router.get("/ping")
def ping():
    return "pong"

@router.get("/count", response_model=CountResponse)
def get_count(db: Session = Depends(get_db)):
    c = db.query(CreditCard).count()
    return {"count": c}

@router.post("/reset", response_model=ResetResponse)
def reset_db(db: Session = Depends(get_db)):
    db.query(CreditCard).delete()
    db.commit()
    return {"msg": "Todos los datos fueron eliminados"}

@router.get("", response_model=list[CreditCardDetail])
def list_cards(current_user: dict = Depends(auth_user), db: Session = Depends(get_db)):
    cards = db.query(CreditCard).filter(CreditCard.userId == current_user["id"]).all()
    return cards

@router.post("", response_model=CreditCardResponse, status_code=status.HTTP_201_CREATED)
def create_card(payload: CreditCardCreate, current_user: dict = Depends(auth_user), db: Session = Depends(get_db)):
    if not payload.cardNumber or not payload.cvv or not payload.expirationDate or not payload.cardHolderName:
        raise HTTPException(status_code=400, detail="Campos faltantes o vacíos")
    
    if is_expired(payload.expirationDate):
        raise HTTPException(status_code=412, detail="Tarjeta vencida")
        
    last_four = payload.cardNumber[-4:]
    if len(last_four) < 4:
        raise HTTPException(status_code=400, detail="Tarjeta inválida")

    exists = db.query(CreditCard).filter(
        CreditCard.userId == current_user["id"],
        CreditCard.lastFourDigits == last_four
    ).first()
    
    if exists:
        raise HTTPException(status_code=409, detail="El usuario ya tiene esta tarjeta almacenada")

    tr_id = str(uuid.uuid4())
    
    tn_payload = {
        "card": {
            "cardNumber": payload.cardNumber,
            "cvv": payload.cvv,
            "expirationDate": payload.expirationDate,
            "cardHolderName": payload.cardHolderName
        },
        "transactionIdentifier": tr_id
    }
    
    # Esto tirará excepción si falla, FastAPI devolverá 400/502
    tn_resp = register_card(tn_payload)
    print(f"TrueNative REGISTRATION RESP: {tn_resp}")
    
    new_card = CreditCard(
        token=tn_resp.get("token", ""),
        userId=current_user["id"],
        lastFourDigits=last_four,
        ruv=tn_resp.get("RUV") or tn_resp.get("ruv") or "",
        issuer=tn_resp.get("issuer", "UNKNOWN"),
        status="POR_VERIFICAR"
    )
    
    db.add(new_card)
    db.commit()
    db.refresh(new_card)
    
    # Send to SQS (Asynchronous background piece)
    publish_card_event(
        card_id=str(new_card.id),
        ruv=new_card.ruv,
        user_id=new_card.userId,
        email=current_user.get("email", ""),
        created_at=new_card.createdAt.isoformat()
    )
    
    return {
        "id": new_card.id,
        "userId": new_card.userId,
        "createdAt": new_card.createdAt
    }
