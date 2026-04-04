from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, ConfigDict
from app.database import Base

# Modelo de SQLAlchemy para la tabla posts
class PostDB(Base):
    __tablename__ = "posts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))  # ID único generado automáticamente
    userId = Column(String, nullable=False, index=True)  # ID del usuario que crea el post
    routeId = Column(String, nullable=False, index=True)  # ID de la ruta asociada al post
    createdAt = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))  # Fecha de creación
    expireAt = Column(DateTime(timezone=True), nullable=False)  # Fecha de expiración del post

# Modelo Pydantic para la API
class Post(BaseModel):
    id: str
    userId: str
    routeId: str
    createdAt: datetime
    expireAt: datetime

    # Permite crear instancias del modelo a partir de atributos de objetos (como instancias de SQLAlchemy)
    model_config = ConfigDict(
        from_attributes=True
    )
