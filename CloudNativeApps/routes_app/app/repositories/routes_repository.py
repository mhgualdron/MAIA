from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete, func
from app import models, schemas
import uuid

class RoutesRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, trayecto: schemas.TrayectoCreate) -> models.Trayecto:
        db_trayecto = models.Trayecto(**trayecto.dict())
        # Let the model generate the UUID default, or we can explicit set it if needed.
        # But our model has default=lambda: str(uuid.uuid4())
        self.db.add(db_trayecto)
        await self.db.commit()
        await self.db.refresh(db_trayecto)
        return db_trayecto

    async def get_by_id(self, id: str) -> models.Trayecto | None:
        result = await self.db.execute(select(models.Trayecto).filter(models.Trayecto.id == id))
        return result.scalars().first()

    async def get_by_flight_id(self, flight_id: str) -> models.Trayecto | None:
        result = await self.db.execute(select(models.Trayecto).filter(models.Trayecto.flightId == flight_id))
        return result.scalars().first()

    async def list_all(self, flight_id: str = None) -> list[models.Trayecto]:
        query = select(models.Trayecto)
        if flight_id:
            query = query.filter(models.Trayecto.flightId == flight_id)
        result = await self.db.execute(query)
        return result.scalars().all()

    async def delete_by_id(self, id: str) -> bool:
        result = await self.db.execute(delete(models.Trayecto).where(models.Trayecto.id == id))
        await self.db.commit()
        return result.rowcount > 0

    async def count(self) -> int:
        result = await self.db.execute(select(func.count()).select_from(models.Trayecto))
        return result.scalar()

    async def reset(self):
        await self.db.execute(delete(models.Trayecto))
        await self.db.commit()
