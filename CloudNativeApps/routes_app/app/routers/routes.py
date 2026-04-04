from fastapi import APIRouter, Depends, HTTPException, status, Header, Response
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app import schemas
from app.repositories.routes_repository import RoutesRepository
from app.services.routes_service import RoutesService
import uuid

router = APIRouter(prefix="/routes", tags=["routes"])

async def get_service(db: AsyncSession = Depends(get_db)) -> RoutesService:
    repository = RoutesRepository(db)
    return RoutesService(repository)

async def verify_token(authorization: str = Header(None)):
    if not authorization:
         # raise HTTPException(status_code=401, detail="Unauthorized")
         pass

@router.post("", response_model=schemas.TrayectoCreatedResponse, status_code=201)
async def create_trayecto(
    trayecto: schemas.TrayectoCreate, 
    service: RoutesService = Depends(get_service)
):
    try:
        new_trayecto = await service.create_trayecto(trayecto)
        return new_trayecto
    except ValueError as e:
        msg = str(e)
        if msg == "Las fechas del trayecto no son válidas":
            return JSONResponse(status_code=412, content={"msg": msg})
        if msg == "El flightId ya existe":
            return Response(status_code=412) # Body N/A
        raise HTTPException(status_code=400, detail=msg)

@router.get("", response_model=list[schemas.TrayectoResponse])
async def list_trayectos(
    flight: str = None, 
    service: RoutesService = Depends(get_service)
):
    return await service.list_trayectos(flight)

@router.get("/count", response_model=schemas.CountResponse)
async def count_trayectos(service: RoutesService = Depends(get_service)):
    count = await service.count_trayectos()
    return {"count": count}

@router.get("/ping")
async def ping():
    return Response(content="pong", media_type="text/plain")

@router.post("/reset")
async def reset_db(service: RoutesService = Depends(get_service)):
    await service.reset_db()
    return {"msg": "Todos los datos fueron eliminados"}

@router.get("/{id}", response_model=schemas.TrayectoResponse)
async def get_trayecto(
    id: str, 
    service: RoutesService = Depends(get_service)
):
    try:
        uuid_obj = uuid.UUID(id)
    except ValueError:
         raise HTTPException(status_code=400, detail="El id no es un valor string con formato uuid")
    
    return await service.get_trayecto(id)

@router.delete("/{id}")
async def delete_trayecto(
    id: str, 
    service: RoutesService = Depends(get_service)
):
    try:
        uuid_obj = uuid.UUID(id)
    except ValueError:
         raise HTTPException(status_code=400, detail="El id no es un valor string con formato uuid")

    await service.delete_trayecto(id)
    return {"msg": "el trayecto fue eliminado"}
