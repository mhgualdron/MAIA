from datetime import datetime, timedelta
from app.models import User

# Test login satisfactorio
def test_login_success(test_client, sample_user):
    payload = {
        "username": sample_user.username,
        "password": "password123"
    }
    response = test_client.post("/users/auth", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "token" in data
    assert "id" in data
    assert data["id"] == sample_user.id

# Test login con contraseña incorrecta
def test_login_wrong_password(test_client, sample_user):
    payload = {
        "username": sample_user.username,
        "password": "wrongpassword"
    }
    response = test_client.post("/users/auth", json=payload)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"

# Test login con usuario no verificado (RF-007)
def test_login_unverified_user(test_client, sample_user, test_db):
    sample_user.status = "POR_VERIFICAR"
    test_db.commit()
    
    payload = {
        "username": sample_user.username,
        "password": "password123"
    }
    response = test_client.post("/users/auth", json=payload)
    assert response.status_code == 401
    assert response.json()["detail"] == "User is not verified"


# Test login con usuario incorrecto
def test_login_wrong_username(test_client):
    payload = {
        "username": "ghost",
        "password": "password"
    }
    response = test_client.post("/users/auth", json=payload)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"

# Test obtener usuario actual
def test_get_me_success(test_client, sample_user, test_db):
    payload = {
        "username": sample_user.username,
        "password": "password123"
    }
    login_res = test_client.post("/users/auth", json=payload)
    token = login_res.json()["token"]
    
    headers = {"Authorization": f"Bearer {token}"}
    response = test_client.get("/users/me", headers=headers)
    assert response.status_code == 200
    assert response.json()["username"] == sample_user.username

# Test obtener usuario actual con token inválido
def test_get_me_invalid_token(test_client):
    headers = {"Authorization": "Bearer invalidtoken"}
    response = test_client.get("/users/me", headers=headers)
    assert response.status_code == 401

# Test obtener usuario actual con token expirado
def test_get_me_expired_token(test_client, sample_user, test_db):
    sample_user.token = "expiredtoken"
    sample_user.expireAt = datetime.utcnow() - timedelta(hours=1)
    test_db.commit()
    
    headers = {"Authorization": "Bearer expiredtoken"}
    response = test_client.get("/users/me", headers=headers)
    assert response.status_code == 401
    assert response.json()["detail"] == "Token expired"

# Test obtener usuario actual sin header
def test_get_me_missing_header(test_client):
    response = test_client.get("/users/me")
    assert response.status_code == 403 
