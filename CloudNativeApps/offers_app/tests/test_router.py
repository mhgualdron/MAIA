
def test_ping(client):
    response = client.get("/offers/ping")
    assert response.status_code == 200


def test_reset(client):
    response = client.post("/offers/reset")
    assert response.status_code == 200


def test_create_offer(client):
    payload = {
        "postId": "11111111-1111-1111-1111-111111111111",
        "userId": "22222222-2222-2222-2222-222222222222",
        "description": "test",
        "size": "LARGE",
        "fragile": False,
        "offer": 100
    }
    response = client.post("/offers", json=payload)
    if response.status_code != 201:
        print(response.json())
    assert response.status_code == 201
    assert "id" in response.json()


def test_count(client):
    response = client.get("/offers/count")
    assert response.status_code == 200

