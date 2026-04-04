from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
from uuid import UUID


# ---------- Base ----------
class OfferBase(BaseModel):
    postId: UUID
    userId: UUID
    description: str = Field(..., max_length=140)
    size: str
    fragile: bool
    offer: int


# ---------- Create ----------
class OfferCreate(OfferBase):
    pass


# ---------- Response ----------
class OfferResponse(OfferBase):
    id: UUID
    createdAt: datetime

    class Config:
        from_attributes = True


# ---------- Count ----------
class OfferCount(BaseModel):
    count: int
