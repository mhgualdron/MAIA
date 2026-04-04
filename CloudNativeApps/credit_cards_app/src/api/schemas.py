from pydantic import BaseModel, ConfigDict
from typing import List
import datetime

class CreditCardCreate(BaseModel):
    cardNumber: str
    cvv: str
    expirationDate: str
    cardHolderName: str

class CreditCardResponse(BaseModel):
    id: str
    userId: str
    createdAt: datetime.datetime

class CreditCardDetail(BaseModel):
    id: str
    token: str
    userId: str
    lastFourDigits: str
    issuer: str
    status: str
    createdAt: datetime.datetime
    updatedAt: datetime.datetime

    model_config = ConfigDict(from_attributes=True)

class CountResponse(BaseModel):
    count: int

class ResetResponse(BaseModel):
    msg: str
