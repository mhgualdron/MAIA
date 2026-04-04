from pydantic import BaseModel, Field, UUID4, validator
from datetime import datetime
from typing import Optional, List

class TrayectoBase(BaseModel):
    flightId: str
    sourceAirportCode: str
    sourceCountry: str
    destinyAirportCode: str
    destinyCountry: str
    bagCost: int
    plannedStartDate: datetime
    plannedEndDate: datetime

class TrayectoCreate(TrayectoBase):
    pass

class TrayectoResponse(TrayectoBase):
    id: UUID4
    createdAt: datetime
    # updatedAt is not strictly required in response per spec but good to have if needed
    
    class Config:
        from_attributes = True

class TrayectoCreatedResponse(BaseModel):
    id: UUID4
    createdAt: datetime

class MessageResponse(BaseModel):
    msg: str

class CountResponse(BaseModel):
    count: int
