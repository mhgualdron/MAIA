from sqlalchemy import Column, String, Integer, Boolean, DateTime, CheckConstraint
from app.database import Base
from datetime import datetime
import uuid

class Offer(Base):
    __tablename__ = "offers"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    postId = Column(String, nullable=False)
    userId = Column(String, nullable=False)
    description = Column(String, nullable=False)
    size = Column(String, nullable=False)
    fragile = Column(Boolean, nullable=False)
    offer = Column(Integer, nullable=False)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("size IN ('LARGE', 'MEDIUM', 'SMALL')", name="check_size_valid"),
    )
