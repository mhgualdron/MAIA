from pydantic import BaseModel
from typing import Union, Optional
from datetime import datetime
from uuid import UUID

class OriginDestiny(BaseModel):
    airportCode: Optional[str] = None
    country: Optional[str] = None

class PostCreateRequest(BaseModel):
    flightId: Optional[str] = None
    expireAt: Optional[str] = None
    plannedStartDate: Optional[str] = None
    plannedEndDate: Optional[str] = None
    origin: Optional[OriginDestiny] = None
    destiny: Optional[OriginDestiny] = None
    bagCost: Optional[Union[int, float]] = None

class RouteResponse(BaseModel):
    id: UUID
    createdAt: str

class PostData(BaseModel):
    id: UUID
    userId: UUID
    createdAt: str
    expireAt: str
    route: RouteResponse

class PostResponse(BaseModel):
    data: PostData
    msg: str
