from fastapi.testclient import TestClient
from app.models import User

# Test ping
def test_ping(test_client):
    response = test_client.get("/users/ping")
    assert response.status_code == 200
    assert response.json() == "pong"

# Test crear usuario
def test_create_user_success(test_client):
    payload = {
        "username": "newuser",
        "email": "new@example.com",
        "password": "securepassword",
        "fullName": "New User"
    }
    response = test_client.post("/users", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert "createdAt" in data

# Test crear usuario con campos incompletos
def test_create_user_missing_fields(test_client):
    payload = {"username": "incomplete"}
    response = test_client.post("/users", json=payload)
    assert response.status_code == 400

# Test crear usuario con usuario duplicado
def test_create_user_duplicate_username(test_client, sample_user):
    payload = {
        "username": sample_user.username,
        "email": "other@example.com",
        "password": "password"
    }
    response = test_client.post("/users", json=payload)
    assert response.status_code == 412
    assert response.json()["detail"] == "El usuario ya existe"

# Test crear usuario con email duplicado
def test_create_user_duplicate_email(test_client, sample_user):
    payload = {
        "username": "otheruser",
        "email": sample_user.email,
        "password": "password"
    }
    response = test_client.post("/users", json=payload)
    assert response.status_code == 412
    assert response.json()["detail"] == "El email ya existe"

# Test actualizar usuario
def test_update_user_success(test_client, sample_user):
    payload = {"fullName": "Updated Name", "status": "VERIFICADO"}
    response = test_client.patch(f"/users/{sample_user.id}", json=payload)
    assert response.status_code == 200
    assert response.json()["msg"] == "el usuario ha sido actualizado"

# Test actualizar usuario no encontrado
def test_update_user_not_found(test_client):
    payload = {"fullName": "Ghost"}
    response = test_client.patch("/users/nonexistent-id", json=payload)
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"

# Test actualizar usuario sin campos
def test_update_user_no_fields(test_client, sample_user):
    response = test_client.patch(f"/users/{sample_user.id}", json={})
    assert response.status_code == 400

# Test obtener cantidad de usuarios
def test_get_count(test_client, sample_user):
    response = test_client.get("/users/count")
    assert response.status_code == 200
    assert response.json()["count"] == 1

# Test resetear usuarios
def test_reset(test_client, sample_user):
    response = test_client.post("/users/reset")
    assert response.status_code == 200
    count_response = test_client.get("/users/count")
    assert count_response.json()["count"] == 0
