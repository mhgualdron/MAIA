from datetime import datetime, timezone
from app import schemas, models
from app.repositories.routes_repository import RoutesRepository
from fastapi import HTTPException, status

class RoutesService:
    def __init__(self, repository: RoutesRepository):
        self.repository = repository

    async def create_trayecto(self, trayecto: schemas.TrayectoCreate) -> models.Trayecto:
        # Validate dates
        now = datetime.now(timezone.utc)
        if trayecto.plannedStartDate >= trayecto.plannedEndDate:
             raise ValueError("Las fechas del trayecto no son válidas")
        
        if trayecto.plannedStartDate < now:
             raise ValueError("Las fechas del trayecto no son válidas")

        # Validate unique flightId
        existing = await self.repository.get_by_flight_id(trayecto.flightId)
        if existing:
            raise ValueError("El flightId ya existe")

        return await self.repository.create(trayecto)

    async def get_trayecto(self, id: str) -> models.Trayecto:
        trayecto = await self.repository.get_by_id(id)
        if not trayecto:
            raise HTTPException(status_code=404, detail="El trayecto con ese id no existe")
        return trayecto

    async def list_trayectos(self, flight_id: str = None) -> list[models.Trayecto]:
        return await self.repository.list_all(flight_id)

    async def delete_trayecto(self, id: str) -> bool:
        deleted = await self.repository.delete_by_id(id)
        if not deleted:
            raise HTTPException(status_code=404, detail="El trayecto con ese id no existe")
        return True

    async def count_trayectos(self) -> int:
        return await self.repository.count()

    async def reset_db(self):
        await self.repository.reset()
