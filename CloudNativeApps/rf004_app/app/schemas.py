from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from uuid import UUID

class OfferCreateRequest(BaseModel):
    description: str
    size: str
    fragile: bool
    offer: float

class OfferData(BaseModel):
    id: UUID
    userId: UUID
    createdAt: datetime
    postId: UUID

class OfferResponse(BaseModel):
    data: OfferData
    msg: str
