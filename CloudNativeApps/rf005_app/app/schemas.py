from pydantic import BaseModel
from typing import List, Optional

class OriginDestinySchema(BaseModel):
    airportCode: str
    country: str

class RouteSchema(BaseModel):
    id: str
    flightId: str
    origin: OriginDestinySchema
    destiny: OriginDestinySchema
    bagCost: int

class OfferSchema(BaseModel):
    id: str
    userId: str
    description: str
    size: str
    fragile: bool
    offer: float
    score: Optional[float] = None
    createdAt: str

class PostDataSchema(BaseModel):
    id: str
    expireAt: str
    route: RouteSchema
    plannedStartDate: str
    plannedEndDate: str
    createdAt: str
    offers: List[OfferSchema]

class RF005Response(BaseModel):
    data: PostDataSchema
