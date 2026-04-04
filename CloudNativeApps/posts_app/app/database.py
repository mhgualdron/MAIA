from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

# URL de la base de datos
DATABASE_URL = settings.database_url

# Base para los modelos (DEBE IR ANTES)
Base = declarative_base()

# Crear engine asíncrono
engine = create_async_engine(
    DATABASE_URL,
    echo=True,
)

# Crear SessionLocal asíncrona
SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Dependency para FastAPI
async def get_db():
    async with SessionLocal() as session:
        yield session