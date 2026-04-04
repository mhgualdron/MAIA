from sqlalchemy import Column, String, Float, DateTime
from app.database import Base
from datetime import datetime
import uuid

class Score(Base):
    __tablename__ = "scores"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    offerId = Column(String, nullable=False, index=True)
    score = Column(Float, nullable=False)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
