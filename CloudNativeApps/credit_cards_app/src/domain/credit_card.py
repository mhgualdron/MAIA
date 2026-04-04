import uuid
import datetime
from sqlalchemy import Column, String, DateTime
from src.core.database import Base

class CreditCard(Base):
    __tablename__ = "credit_cards"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    token = Column(String(256), nullable=False)
    userId = Column(String(36), nullable=False)
    lastFourDigits = Column(String(4), nullable=False)
    ruv = Column(String(256), nullable=False)
    issuer = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False, default="POR_VERIFICAR")
    createdAt = Column(DateTime, default=datetime.datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
