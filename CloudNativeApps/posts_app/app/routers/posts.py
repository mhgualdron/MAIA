from fastapi import APIRouter, Depends, HTTPException, Query, Response
from uuid import UUID, uuid4
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
from fastapi.responses import JSONResponse

from app.database import get_db
from app.schemas import (
    PostCreate,
    PostResponse,
    PostCreatedResponse,
    CountResponse,
    MessageResponse,
)
from app.services.posts_service import PostsService
from app.repositories.posts_repository import PostsRepository
from app.models import Post

# Router para los endpoints relacionados con posts
router = APIRouter(prefix="/posts", tags=["posts"])

# Dependency que provee la instancia del servicio de posts
async def get_service(db: AsyncSession = Depends(get_db)) -> PostsService:
    repository = PostsRepository(db)
    return PostsService(repository)

# Endpoint simple para verificar que el servicio está activo
@router.get("/ping")
async def ping():
    return Response(content="pong", media_type="text/plain")

# Endpoint para crear un nuevo post
@router.post("", response_model=PostResponse, status_code=201)
async def create_post(
    post: PostCreate,
    service: PostsService = Depends(get_service),
):
    # Asegura que expireAt tenga timezone
    expire_at_aware = post.expireAt
    if expire_at_aware.tzinfo is None:
        expire_at_aware = expire_at_aware.replace(tzinfo=timezone.utc)

    # Validación: la fecha de expiración no puede ser pasada
    if expire_at_aware <= datetime.now(timezone.utc):
        return JSONResponse(
            status_code=412,
            content={"msg": 'La fecha expiración no es válida'}
        )

    # Validación: userId y routeId deben ser UUID válidos y no vacíos
    for field_name, value in [("userId", post.userId), ("routeId", post.routeId)]:
        if not value:
            raise HTTPException(status_code=400)
        try:
            UUID(value)
        except ValueError:
            raise HTTPException(status_code=400)

    # Crear objeto de dominio para el post
    domain_post = Post(
        id=str(uuid4()),
        routeId=post.routeId,
        userId=post.userId,
        expireAt=expire_at_aware,
        createdAt=datetime.now(timezone.utc),
    )

    # Guardar en la base de datos
    created_post = await service.create_post(domain_post)

    # Retornar la respuesta normal
    return created_post

# Endpoint para listar posts con filtros opcionales
@router.get("", response_model=List[PostResponse])
async def list_posts(
    expire: Optional[bool] = Query(None),
    route: Optional[str] = Query(None),
    owner: Optional[str] = Query(None),
    service: PostsService = Depends(get_service),
):
    return await service.list_posts(expire=expire, route=route, owner=owner)

# Endpoint para obtener la cantidad total de posts
@router.get("/count", response_model=CountResponse)
async def count_posts(service: PostsService = Depends(get_service)):
    return {"count": await service.count_posts()}

# Endpoint para resetear todos los posts (uso principal: testing)
@router.post("/reset", response_model=MessageResponse)
async def reset_posts(service: PostsService = Depends(get_service)):
    await service.reset_posts()
    return {"msg": "Todos los posts fueron eliminados"}

# Endpoint para obtener un post por ID
@router.get("/{post_id}", response_model=PostResponse)
async def get_post(post_id: str, service: PostsService = Depends(get_service)):
    # Validación de formato UUID
    try:
        UUID(post_id)
    except ValueError:
        raise HTTPException(status_code=400)

    post = await service.get_post(post_id)
    if not post:
        raise HTTPException(status_code=404)
    return post

# Endpoint para eliminar un post por ID
@router.delete("/{post_id}")
async def delete_post(post_id: str, service: PostsService = Depends(get_service)):
    # Validar que sea un UUID válido
    try:
        UUID(post_id)
    except ValueError:
        raise HTTPException(status_code=400)

    # Llamar al service para eliminar el post
    await service.delete_post(post_id)
    return {"msg": "la publicación fue eliminada"}
