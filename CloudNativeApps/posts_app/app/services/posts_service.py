from datetime import datetime, timezone
from uuid import UUID
from typing import List, Optional

from app.models import Post
from app.repositories.posts_repository import PostsRepository
from fastapi import HTTPException

# Servicio que maneja la lógica de negocio de los posts
class PostsService:
    def __init__(self, repository: PostsRepository):
        self.repository = repository  # Repositorio para acceder a la base de datos

    # Crear un nuevo post
    async def create_post(self, post: Post) -> Post:
        return await self.repository.create(post)

    # Obtener un post por su ID
    async def get_post(self, post_id: str) -> Post:
        post = await self.repository.get_by_id(post_id)
        if not post:
            raise HTTPException(status_code=404, detail="Post no encontrado")
        return post

    # Listar posts con filtros opcionales de expiración, ruta y propietario
    async def list_posts(
            self,
            expire: Optional[bool] = None,
            route: Optional[str] = None,
            owner: Optional[str] = None,
    ) -> List[Post]:
        posts = await self.repository.list_all()
        now = datetime.now(timezone.utc)  # siempre aware en UTC

        def make_aware(dt: datetime) -> datetime:
            """Convierte un datetime naive a UTC-aware."""
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        # Filtrar por posts expirados o activos
        if expire is not None:
            if expire:  # queremos expirados
                posts = [p for p in posts if make_aware(p.expireAt) <= now]
            else:  # queremos activos
                posts = [p for p in posts if make_aware(p.expireAt) > now]

        # Filtrar por route_id si se especifica
        if route:
            posts = [p for p in posts if p.routeId == route]

        # Filtrar por owner si se especifica
        if owner:
            posts = [p for p in posts if p.userId == owner]

        return posts


    # Eliminar un post por su ID
    async def delete_post(self, post_id: str) -> None:
        deleted = await self.repository.delete(post_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Post no encontrado")

    # Contar el total de posts
    async def count_posts(self) -> int:
        return await self.repository.count()

    # Resetear todos los posts (usado principalmente para testing)
    async def reset_posts(self) -> None:
        await self.repository.reset()