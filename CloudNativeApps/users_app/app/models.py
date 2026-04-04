from sqlalchemy import Column, String, DateTime
from app.database import Base
from datetime import datetime
import uuid

class User(Base):
    __tablename__ = "users"

    # Campos obligatorios
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    salt = Column(String, nullable=False)
    token = Column(String, nullable=True)
    status = Column(String, nullable=False, default="POR_VERIFICAR")
    expireAt = Column(DateTime, nullable=True)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    updatedAt = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Campos opcionales
    phoneNumber = Column(String, nullable=True)
    dni = Column(String, nullable=True)
    fullName = Column(String, nullable=True)
