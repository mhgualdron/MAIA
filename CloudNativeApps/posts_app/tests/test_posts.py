import pytest
from datetime import datetime, timedelta, timezone
import uuid

# Test endpoint de healthcheck
@pytest.mark.asyncio
async def test_ping(client):
    response = await client.get("/posts/ping")
    assert response.status_code == 200
    assert response.text == "pong"

# Test creación de un post
@pytest.mark.asyncio
async def test_create_post(client):
    payload = {
        "userId": str(uuid.uuid4()),
        "routeId": str(uuid.uuid4()),
        "expireAt": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    }

    response = await client.post("/posts", json=payload)

    assert response.status_code == 201
    data = response.json()

    # Verificar que el backend haya generado un ID
    assert "id" in data
    assert isinstance(data["id"], str)

# Test obtener un post existente
@pytest.mark.asyncio
async def test_get_post(client):
    payload = {
        "userId": str(uuid.uuid4()),
        "routeId": str(uuid.uuid4()),
        "expireAt": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    }

    # Crear post primero
    create_response = await client.post("/posts", json=payload)
    created_post = create_response.json()
    post_id = created_post["id"]

    # Obtener el post por ID
    response = await client.get(f"/posts/{post_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == post_id

# Test obtener un post que no existe
@pytest.mark.asyncio
async def test_get_post_not_found(client):
    random_id = str(uuid.uuid4())
    response = await client.get(f"/posts/{random_id}")
    assert response.status_code == 404

# Test eliminar un post existente
@pytest.mark.asyncio
async def test_delete_post(client):
    payload = {
        "userId": str(uuid.uuid4()),
        "routeId": str(uuid.uuid4()),
        "expireAt": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    }

    # Crear post primero
    create_response = await client.post("/posts", json=payload)
    post_id = create_response.json()["id"]

    # Eliminar el post
    response = await client.delete(f"/posts/{post_id}")
    assert response.status_code == 200

# Test contar el número total de posts
@pytest.mark.asyncio
async def test_count_posts(client):
    response = await client.get("/posts/count")
    assert response.status_code == 200
    assert "count" in response.json()

# Test listar posts sin filtros
@pytest.mark.asyncio
async def test_list_posts(client):
    response = await client.get("/posts")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

# Test listar posts con filtros
@pytest.mark.asyncio
async def test_list_posts_filter(client):
    # Crear 2 publicaciones con distintos owners y rutas
    post1_id = str(uuid.uuid4())
    post2_id = str(uuid.uuid4())
    payload1 = {
        "id": post1_id,
        "userId": "userA",
        "routeId": "route1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "expireAt": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    }
    payload2 = {
        "id": post2_id,
        "userId": "userB",
        "routeId": "route2",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "expireAt": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    }
    await client.post("/posts", json=payload1)
    await client.post("/posts", json=payload2)

    # Filtrar por owner
    response = await client.get("/posts", params={"owner": "userA"})
    assert response.status_code == 200
    data = response.json()
    assert all(p["userId"] == "userA" for p in data)

# Test obtener un post con un UUID inválido
@pytest.mark.asyncio
async def test_get_post_invalid_uuid(client):
    response = await client.get("/posts/invalid-uuid-12345")
    assert response.status_code == 400
    assert "detail" in response.json()