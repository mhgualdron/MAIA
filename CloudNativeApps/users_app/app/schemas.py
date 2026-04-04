from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from uuid import UUID

class UserBase(BaseModel):
    username: str
    email: EmailStr
    fullName: Optional[str] = None
    phoneNumber: Optional[str] = None
    dni: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    status: Optional[str] = None
    fullName: Optional[str] = None
    phoneNumber: Optional[str] = None
    dni: Optional[str] = None

class UserResponse(UserBase):
    id: UUID
    status: str
    createdAt: datetime
    updatedAt: datetime

    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    id: UUID
    token: str
    expireAt: datetime

from typing import Any

class UserVerificationCallback(BaseModel):
    RUV: Any = None
    userIdentifier: Any = None
    createdAt: Any = None
    status: Any = None
    score: Any = None
    verifyToken: Any = None
