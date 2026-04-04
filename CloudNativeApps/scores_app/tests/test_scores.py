from fastapi.testclient import TestClient
from app.main import app
from app.database import Base, engine, get_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest
import uuid

# SQLite for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine_test = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine_test)
    yield
    Base.metadata.drop_all(bind=engine_test)

def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.text == '"pong"'

def test_create_and_get_score():
    offer_id = str(uuid.uuid4())
    payload = {"offerId": offer_id, "score": 150.5}
    response = client.post("/scores", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["offerId"] == offer_id
    assert data["score"] == 150.5

    # Get by offer ID
    response = client.get(f"/scores/{offer_id}")
    assert response.status_code == 200
    assert response.json()["score"] == 150.5

def test_count_and_reset():
    client.post("/scores", json={"offerId": str(uuid.uuid4()), "score": 10.0})
    response = client.get("/scores/count")
    assert response.json()["count"] == 1

    client.post("/scores/reset")
    response = client.get("/scores/count")
    assert response.json()["count"] == 0
