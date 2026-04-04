import pytest
import respx
from httpx import Response
from app.config import settings

@pytest.mark.asyncio
async def test_ping(client):
    response = await client.get("/ping")
    assert response.status_code == 200
    assert response.json() == "pong"

@pytest.mark.asyncio
async def test_rf005_unauthorized(client):
    response = await client.get("/rf005/posts/123")
    assert response.status_code == 401

@pytest.mark.asyncio
@respx.mock
async def test_rf005_forbidden(client):
    respx.get(f"{settings.users_service_url}/users/me").mock(return_value=Response(200, json={"id": "user1"}))
    respx.get(f"{settings.posts_service_url}/posts/123").mock(return_value=Response(200, json={"id": "123", "userId": "user2"}))
    
    response = await client.get("/rf005/posts/123", headers={"Authorization": "Bearer token"})
    assert response.status_code == 403

@pytest.mark.asyncio
@respx.mock
async def test_rf005_post_not_found(client):
    respx.get(f"{settings.users_service_url}/users/me").mock(return_value=Response(200, json={"id": "user1"}))
    respx.get(f"{settings.posts_service_url}/posts/123").mock(return_value=Response(404))
    
    response = await client.get("/rf005/posts/123", headers={"Authorization": "Bearer token"})
    assert response.status_code == 404

@pytest.mark.asyncio
@respx.mock
async def test_rf005_success_with_scores(client):
    # Mock Users
    respx.get(f"{settings.users_service_url}/users/me").mock(return_value=Response(200, json={"id": "user1"}))
    
    # Mock Posts
    post_data = {
        "id": "123",
        "userId": "user1",
        "expireAt": "2026-12-31T23:59:59Z",
        "routeId": "route123", # mapped to "id" internally in Route schema by API? Wait, the API schema says "route: {id, flightId, ...}"
        "createdAt": "2026-01-01T00:00:00Z"
    }
    respx.get(f"{settings.posts_service_url}/posts/123").mock(return_value=Response(200, json=post_data))
    
    # Mock Routes
    route_data = {
        "id": "route123",
        "flightId": "flight123",
        "origin": {"airportCode": "BOG", "country": "Colombia"},
        "destiny": {"airportCode": "MIA", "country": "USA"},
        "bagCost": 50,
        "plannedStartDate": "2026-06-01T00:00:00Z",
        "plannedEndDate": "2026-06-02T00:00:00Z"
    }
    respx.get(f"{settings.routes_service_url}/routes/route123").mock(return_value=Response(200, json={
        "id": "route123",
        "flightId": "flight123",
        "sourceAirportCode": "BOG",
        "sourceCountry": "Colombia",
        "destinyAirportCode": "MIA",
        "destinyCountry": "USA",
        "bagCost": 50,
        "plannedStartDate": "2026-06-01T00:00:00Z",
        "plannedEndDate": "2026-06-02T00:00:00Z"
    }))
    
    # Mock Offers
    offers_data = [
        {"id": "o1", "userId": "u2", "description": "Paquete 1", "size": "LARGE", "fragile": False, "offer": 100, "createdAt": "2026-01-02T00:00:00Z"},
        {"id": "o2", "userId": "u3", "description": "Paquete 2", "size": "SMALL", "fragile": True, "offer": 20, "createdAt": "2026-01-02T01:00:00Z"},
        {"id": "o3", "userId": "u4", "description": "Paquete 3", "size": "MEDIUM", "fragile": False, "offer": 50, "createdAt": "2026-01-02T02:00:00Z"}
    ]
    respx.get(f"{settings.offers_service_url}/offers").mock(return_value=Response(200, json=offers_data))
    
    # Mock Scores: o1 -> 50, o2 -> not found (null), o3 -> 25
    respx.get(f"{settings.scores_service_url}/scores/o1").mock(return_value=Response(200, json={"score": 50}))
    respx.get(f"{settings.scores_service_url}/scores/o2").mock(return_value=Response(404))
    respx.get(f"{settings.scores_service_url}/scores/o3").mock(return_value=Response(200, json={"score": 25}))
    
    response = await client.get("/rf005/posts/123", headers={"Authorization": "Bearer token"})
    
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["id"] == "123"
    assert len(data["offers"]) == 3
    # Offers should be sorted by score descending: o1 (50) -> o3 (25) -> o2 (null)
    assert data["offers"][0]["id"] == "o1"
    assert data["offers"][0]["score"] == 50
    assert data["offers"][1]["id"] == "o3"
    assert data["offers"][1]["score"] == 25
    assert data["offers"][2]["id"] == "o2"
    assert data["offers"][2]["score"] is None

@pytest.mark.asyncio
@respx.mock
async def test_rf005_service_unavailable(client):
    respx.get(f"{settings.users_service_url}/users/me").mock(return_value=Response(503))
    response = await client.get("/rf005/posts/123", headers={"Authorization": "Bearer token"})
    assert response.status_code == 503
