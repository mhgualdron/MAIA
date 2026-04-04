from sqlalchemy import Column, String, Integer, DateTime
from app.database import Base
from datetime import datetime, timezone
import uuid

class Trayecto(Base):
    __tablename__ = "routes"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    flightId = Column(String, unique=True, index=True, nullable=False)
    sourceAirportCode = Column(String, nullable=False)
    sourceCountry = Column(String, nullable=False)
    destinyAirportCode = Column(String, nullable=False)
    destinyCountry = Column(String, nullable=False)
    bagCost = Column(Integer, nullable=False)
    # Use timezone=True to mapping to TIMESTAMPTZ
    plannedStartDate = Column(DateTime(timezone=True), nullable=False)
    plannedEndDate = Column(DateTime(timezone=True), nullable=False)
    createdAt = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updatedAt = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
