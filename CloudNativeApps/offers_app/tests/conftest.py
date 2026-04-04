import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from app.database import Base, get_db
from app.main import app
from unittest.mock import MagicMock

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Fixture for test database session
@pytest.fixture(scope="function")
def test_db():
    # Create tables in the in-memory SQLite database
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

# Fixture for TestClient
@pytest.fixture(scope="function")
def client(test_db):
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    # Mock the lifespan or startup event to avoid real DB connection attempt
    # We patch create_all to do nothing during test startup if it's called
    with pytest.MonkeyPatch.context() as m:
        m.setattr("app.database.Base.metadata.create_all", MagicMock())
        with TestClient(app) as c:
            yield c
            
    app.dependency_overrides.clear()
