import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.database import Base, get_db
from app.main import app
from httpx import AsyncClient, ASGITransport
from app.config import settings

# Use internal SQLite for tests to avoid external DB dependency in CI
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest_asyncio.fixture(scope="session")
async def engine():
    # StaticPool is required for in-memory SQLite to share connection across the session
    engine = create_async_engine(TEST_DATABASE_URL, poolclass=StaticPool, echo=False)
    yield engine
    await engine.dispose()

@pytest_asyncio.fixture(scope="function")
async def async_session(engine):
    # Match users_app strategy: Create tables before test, Drop after test
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Setup session
    TestingSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    
    async with TestingSessionLocal() as session:
        yield session
    
    # Teardown: Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture(scope="function")
async def client(async_session):
    async def override_get_db():
        yield async_session

    app.dependency_overrides[get_db] = override_get_db
    # Force app to use the lifespan logic? 
    # With AsyncClient and ASGITransport, lifespan is usually triggered if using `LifespanManager` or manual.
    # But for us, connection is injected via override, so app's internal engine usage is bypassed for request handling.
    # However, `lifespan` in `main.py` creates tables too. We should verify that doesn't conflict.
    # Since we override `get_db`, the endpoints use our test session.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
