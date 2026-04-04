from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete, func
from datetime import datetime, timezone
import uuid

from app.models import PostDB
from app.schemas import PostCreate

class PostsRepository:
    def __init__(self, db: AsyncSession):
        self.db = db  # Sesión de base de datos asíncrona

    # Crear un nuevo post
    async def create(self, post: PostCreate) -> PostDB:
        db_post = PostDB(
            id=str(uuid.uuid4()),  # Genera un ID único
            userId=post.userId,
            routeId=post.routeId,
            createdAt=datetime.now(timezone.utc),  # Fecha de creación en UTC
            expireAt=post.expireAt
        )
        self.db.add(db_post)
        await self.db.commit()  # Guarda el post en la base de datos
        await self.db.refresh(db_post)  # Refresca el objeto con los datos persistidos
        return db_post

    # Obtener un post por su ID
    async def get_by_id(self, post_id: str) -> PostDB | None:
        result = await self.db.execute(select(PostDB).filter(PostDB.id == post_id))
        return result.scalars().first()

    # Listar posts con filtros opcionales
    async def list_all(
        self,
        expire: bool | None = None,
        route_id: str | None = None,
        user_id: str | None = None
    ) -> list[PostDB]:
        query = select(PostDB)
        now = datetime.now(timezone.utc)

        if expire is not None:
            if expire:
                query = query.filter(PostDB.expireAt <= now)  # Solo posts expirados
            else:
                query = query.filter(PostDB.expireAt > now)  # Solo posts activos

        if route_id:
            query = query.filter(PostDB.routeId == route_id)

        if user_id:
            query = query.filter(PostDB.userId == user_id)

        result = await self.db.execute(query)
        return result.scalars().all()

    # Eliminar un post por su ID
    async def delete(self, post_id: str) -> bool:
        result = await self.db.execute(delete(PostDB).where(PostDB.id == post_id))
        await self.db.commit()
        return result.rowcount > 0

    # Contar el total de posts
    async def count(self) -> int:
        result = await self.db.execute(select(func.count()).select_from(PostDB))
        return result.scalar()

    # Resetear la tabla de posts (usado para testing)
    async def reset(self):
        await self.db.execute(delete(PostDB))
        await self.db.commit()