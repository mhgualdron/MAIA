import pytest
from datetime import datetime, timedelta, timezone
import uuid

@pytest.mark.asyncio
async def test_ping(client):
    response = await client.get("/routes/ping")
    assert response.status_code == 200
    assert response.text == "pong"

@pytest.mark.asyncio
async def test_create_trayecto(client):
    flight_id = f"AA{uuid.uuid4().hex[:4]}"
    payload = {
        "flightId": flight_id,
        "sourceAirportCode": "BOG",
        "sourceCountry": "Colombia",
        "destinyAirportCode": "MIA",
        "destinyCountry": "USA",
        "bagCost": 50,
        "plannedStartDate": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        "plannedEndDate": (datetime.now(timezone.utc) + timedelta(days=1, hours=5)).isoformat()
    }
    response = await client.post("/routes", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert "createdAt" in data

@pytest.mark.asyncio
async def test_create_trayecto_invalid_dates(client):
    flight_id = f"AA{uuid.uuid4().hex[:4]}"
    # End before Start
    payload = {
        "flightId": flight_id,
        "sourceAirportCode": "BOG",
        "sourceCountry": "Colombia",
        "destinyAirportCode": "MIA",
        "destinyCountry": "USA",
        "bagCost": 50,
        "plannedStartDate": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
        "plannedEndDate": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    }
    response = await client.post("/routes", json=payload)
    assert response.status_code == 412
    assert response.json()["msg"] == "Las fechas del trayecto no son válidas"

@pytest.mark.asyncio
async def test_create_trayecto_duplicate_flight(client):
    flight_id = f"AA{uuid.uuid4().hex[:4]}"
    payload = {
        "flightId": flight_id,
        "sourceAirportCode": "BOG",
        "sourceCountry": "Colombia",
        "destinyAirportCode": "MIA",
        "destinyCountry": "USA",
        "bagCost": 50,
        "plannedStartDate": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        "plannedEndDate": (datetime.now(timezone.utc) + timedelta(days=1, hours=5)).isoformat()
    }
    await client.post("/routes", json=payload)
    
    # Try creating again
    response = await client.post("/routes", json=payload)
    assert response.status_code == 412
    # Spec says N/A body for duplicate.
    assert response.content == b"" or response.content == b"null" or len(response.content) == 0

@pytest.mark.asyncio
async def test_get_trayecto(client):
    # Create one first
    flight_id = f"AA{uuid.uuid4().hex[:4]}"
    payload = {
        "flightId": flight_id,
        "sourceAirportCode": "MAD",
        "sourceCountry": "Spain",
        "destinyAirportCode": "CDG",
        "destinyCountry": "France",
        "bagCost": 30,
        "plannedStartDate": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        "plannedEndDate": (datetime.now(timezone.utc) + timedelta(days=1, hours=2)).isoformat()
    }
    create_res = await client.post("/routes", json=payload)
    trayecto_id = create_res.json()["id"]

    # Get it
    response = await client.get(f"/routes/{trayecto_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["flightId"] == flight_id

@pytest.mark.asyncio
async def test_get_trayecto_not_found(client):
    random_id = str(uuid.uuid4())
    response = await client.get(f"/routes/{random_id}")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_list_trayectos(client):
    # Ensure clean slate or just count
    response = await client.get("/routes")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

@pytest.mark.asyncio
async def test_delete_trayecto(client):
    # Create one
    flight_id = f"DEL{uuid.uuid4().hex[:4]}"
    payload = {
        "flightId": flight_id,
        "sourceAirportCode": "A",
        "sourceCountry": "B",
        "destinyAirportCode": "C",
        "destinyCountry": "D",
        "bagCost": 10,
        "plannedStartDate": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        "plannedEndDate": (datetime.now(timezone.utc) + timedelta(days=1, hours=1)).isoformat()
    }
    create_res = await client.post("/routes", json=payload)
    trayecto_id = create_res.json()["id"]

    # Delete
    response = await client.delete(f"/routes/{trayecto_id}")
    assert response.status_code == 200
    assert response.json()["msg"] == "el trayecto fue eliminado"

    # Verify gone
    get_res = await client.get(f"/routes/{trayecto_id}")
    assert get_res.status_code == 404

@pytest.mark.asyncio
async def test_count_trayectos(client):
    response = await client.get("/routes/count")
    assert response.status_code == 200
    assert "count" in response.json()

@pytest.mark.asyncio
async def test_list_trayectos_filter_by_flight(client):
    # Create two routes
    flight1 = f"F1{uuid.uuid4().hex[:4]}"
    flight2 = f"F2{uuid.uuid4().hex[:4]}"
    
    # helper
    async def create(fid):
        payload = {
            "flightId": fid,
            "sourceAirportCode": "A", "sourceCountry": "B", "destinyAirportCode": "C", "destinyCountry": "D", "bagCost": 10,
            "plannedStartDate": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            "plannedEndDate": (datetime.now(timezone.utc) + timedelta(days=1, hours=1)).isoformat()
        }
        await client.post("/routes", json=payload)

    await create(flight1)
    await create(flight2)

    # Filter by flight1
    response = await client.get(f"/routes?flight={flight1}")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    # Verify all returned have that flightId
    for item in data:
        assert item["flightId"] == flight1

@pytest.mark.asyncio
async def test_get_trayecto_invalid_uuid(client):
    response = await client.get("/routes/invalid-uuid-12345")
    assert response.status_code == 400
    # Spec says 400 The id is not a string with uuid format.
    # Our impl returns {"detail": "El id no es un valor string con formato uuid"}
    assert response.json()["detail"] == "El id no es un valor string con formato uuid"
