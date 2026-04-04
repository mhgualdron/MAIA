from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

# Clase base que define los campos comunes de un post
class PostBase(BaseModel):

    userId : str
    routeId: str
    expireAt: datetime

class PostCreate(PostBase):
    pass

class PostResponse(PostBase):
    id: str
    createdAt: datetime

    model_config = ConfigDict(
        from_attributes=True
    )

class PostCreatedResponse(BaseModel):
    id: str
    createdAt: datetime

class CountResponse(BaseModel):
    count: int

class MessageResponse(BaseModel):
    msg: str
