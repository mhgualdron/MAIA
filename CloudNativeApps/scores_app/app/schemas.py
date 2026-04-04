from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from uuid import UUID

class ScoreBase(BaseModel):
    offerId: UUID
    score: float

class ScoreCreate(ScoreBase):
    pass

class ScoreResponse(ScoreBase):
    id: UUID
    createdAt: datetime

    class Config:
        from_attributes = True

class ScoreCount(BaseModel):
    count: int
