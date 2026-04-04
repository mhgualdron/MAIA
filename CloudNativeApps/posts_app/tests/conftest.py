import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from httpx import AsyncClient, ASGITransport

from app import database
from app.main import app

# URL de base de datos de prueba en memoria
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Fixture para crear y destruir el engine de base de datos de pruebas
@pytest_asyncio.fixture(scope="session")
async def engine():
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    # Crear todas las tablas al inicio de la sesión de pruebas
    async with engine.begin() as conn:
        await conn.run_sync(database.Base.metadata.create_all)

    yield engine

    # Eliminar todas las tablas al final de la sesión de pruebas
    async with engine.begin() as conn:
        await conn.run_sync(database.Base.metadata.drop_all)

    # Cerrar el engine
    await engine.dispose()

# Fixture para proporcionar una sesión de base de datos por función de test
@pytest_asyncio.fixture(scope="function")
async def async_session(engine):
    TestingSessionLocal = sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with TestingSessionLocal() as session:
        yield session

# Fixture para proporcionar un cliente HTTP asíncrono para probar los endpoints de FastAPI
@pytest_asyncio.fixture(scope="function")
async def client(async_session):
    # Sobrescribir la dependencia get_db para usar la sesión de prueba
    async def override_get_db():
        yield async_session

    app.dependency_overrides[database.get_db] = override_get_db

    # Crear cliente HTTP para realizar requests a la app en memoria
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    # Limpiar las dependencias sobreescritas
    app.dependency_overrides.clear()