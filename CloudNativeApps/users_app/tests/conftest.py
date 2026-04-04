import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from app.database import Base, get_db
from app.main import app
from app.utils import hash_password


SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Fixture para la base de datos
@pytest.fixture(scope="function")
def test_db():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

# Fixture para el cliente de pruebas
@pytest.fixture(scope="function")
def test_client(test_db):
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    # Mockear la creación de tablas en el lifespan para evitar conexión a DB real
    from unittest.mock import patch
    with patch("app.main.Base.metadata.create_all"):
        with TestClient(app) as client:
            yield client
    app.dependency_overrides.clear()

# Fixture para un usuario de prueba
@pytest.fixture(scope="function")
def sample_user(test_db):
    from app.models import User
    hashed_pwd, salt = hash_password("password123")
    user = User(
        username="testuser",
        email="test@example.com",
        password=hashed_pwd,
        salt=salt,
        phoneNumber="123456789",
        dni="12345678",
        status="VERIFICADO"
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user

